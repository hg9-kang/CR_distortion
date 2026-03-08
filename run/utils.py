import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import cv2
import torch.nn.functional as F


import albumentations as A
import random
import math


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


def collate_fn(batch):
    x, y = zip(*batch)
    x = np.stack(x)
    y = torch.LongTensor(np.stack(y))
    return x, y


use_amp = True
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
np.bool = bool

def augmix(x, general_transform, distort_transform, normalize, alpha=1):
    [x_1, x_2, x_3] = [np.stack([distort_transform(image=x_)["image"] for x_ in x]) for _ in range(3)]
    # x_gen = np.stack([general_transform(image=x_)["image"] for x_ in x])
    x_gen = x
    ws = np.random.dirichlet([alpha] * 3)
    m = np.random.beta(alpha, alpha)
    mix = np.zeros_like(x, dtype=np.float32)
    for i in range(len(x)):
        mix_ = ws[0] * x_1[i] + ws[1] * x_2[i] + ws[2] * x_3[i]
        mix[i] = (1 - m) * x_gen[i] + m * mix_
    mix = mix.astype(np.uint8)
    mix = torch.stack([normalize(image=m)["image"] for m in mix])
    return mix

def train(
    model,
    dataloader_train,
    optimizer,
    loss_fn_cls,
    general_transform,
    distort_transform,
    normalize,
    epoch,
    writer,
    args,
):
    model.train()
    total_loss, total_loss_cls, total_loss_cr = 0, 0, 0
    for x, y in dataloader_train:
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
            x_base = np.stack([general_transform(image=x)["image"] for x in x])
            x_general = torch.stack(
                [normalize(image=x_)["image"] for x_ in x_base]
            ).cuda()
            y_pred = model(x_general)
            loss = loss_fn_cls(y_pred, y.cuda())
            total_loss_cls += loss.item()
            if args.method == "CR":
                x_distorted_list = [torch.stack(
                            [
                                normalize(image=x_)["image"]
                                for x_ in np.stack(
                                    [distort_transform(image=x_)["image"] for x_ in x_base]
                                )
                            ]
                        ).cuda() for _ in range(args.num_views)]

                y_pred_distorted_list = [model(x_distorted) for x_distorted in x_distorted_list]
                # ========== Consistency Regularization Loss (CR Loss) ==========
                if args.cr_loss == "ce":
                    # 1. Clean 이미지의 예측값에서 확률 분포를 직접 추출 (Soft Label)
                    # 별도의 argmax 없이 확률 분포(p_clean)를 그대로 타겟으로 사용합니다.
                    p_clean = y_pred.softmax(dim=1) 

                    loss_cr = 0
                    for i in range(args.num_views):
                        # 2. PyTorch 1.10+ 부터는 cross_entropy의 target 인자에 
                        # 클래스 인덱스가 아닌 '확률 분포'를 직접 넣을 수 있습니다.
                        loss_cr += F.cross_entropy(
                            y_pred_distorted_list[i],  # Logits (Model Output)
                            p_clean,                   # Soft Labels (Target Distribution)
                            reduction="mean"
                        )
                    loss_cr /= args.num_views


                elif args.cr_loss == "kl":
                    # KL( p_clean || p_distorted )
                    p_clean = y_pred.softmax(dim=1)

                    loss_cr = 0
                    for i in range(args.num_views):
                        log_p_dist = y_pred_distorted_list[i].log_softmax(dim=1)
                        loss_cr += F.kl_div(
                            log_p_dist,            # log q(x)
                            p_clean,               # p(x)
                            reduction="batchmean"
                        )
                    loss_cr /= args.num_views


                elif args.cr_loss == "js":
                    # Jensen-Shannon Divergence
                    p_clean = y_pred.softmax(dim=1)

                    loss_cr = 0
                    for i in range(args.num_views):
                        p_dist = y_pred_distorted_list[i].softmax(dim=1)
                        m = 0.5 * (p_clean + p_dist)

                        loss_cr += 0.5 * (
                            F.kl_div(p_clean.log(), m, reduction="batchmean")
                            + F.kl_div(p_dist.log(), m, reduction="batchmean")
                        )
                    loss_cr /= args.num_views


                elif args.cr_loss == "l2":
                    # L2 distance between softmax distributions
                    p_clean = y_pred.softmax(dim=1)

                    loss_cr = 0
                    for i in range(args.num_views):
                        p_dist = y_pred_distorted_list[i].softmax(dim=1)
                        loss_cr += F.mse_loss(p_dist, p_clean)
                    loss_cr /= args.num_views


                # Add to main loss
                loss = loss + args.lambda_ * loss_cr
                total_loss_cr += loss_cr.item()
            
            elif args.method == 'augmix':
                mix_1 = augmix(x, general_transform, distort_transform, normalize)
                mix_2 = augmix(x, general_transform, distort_transform, normalize)
                p_mix_1 = model(mix_1.cuda()).softmax(dim=1)
                p_mix_2 = model(mix_2.cuda()).softmax(dim=1)
                p_clean = y_pred.softmax(dim=1)
                p_mixture = torch.clamp((p_mix_1 + p_mix_2 + p_clean) / 3, 1e-7, 1).log()
                loss_cr = (F.kl_div(p_mixture, p_clean, reduction="batchmean") + F.kl_div(p_mixture, p_mix_1, reduction="batchmean") + F.kl_div(p_mixture, p_mix_2, reduction="batchmean")) / 3
                loss = loss + 12*loss_cr
        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # total_loss += loss.item()
    train_loss = total_loss / len(dataloader_train)
    train_loss_cls = total_loss_cls / len(dataloader_train)
    train_loss_cr = total_loss_cr / len(dataloader_train)

    writer.add_scalar("Train/loss", train_loss, epoch)
    writer.add_scalar("Train/loss_cls", train_loss_cls, epoch)
    writer.add_scalar("Train/loss_cr", train_loss_cr, epoch)

    return train_loss



def validate(
    model,
    dataloader_val,
    loss_fn_cls,
    general_transform,
    distort_transform,
    normalize,
    epoch,
    writer,
    args,
):
    model.eval()
    total_loss, total_loss_cls, total_loss_cr, total_pred = 0, 0, 0, []
    with torch.no_grad():
        for x, y in dataloader_val:
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                x_base = np.stack([general_transform(image=x)["image"] for x in x])
                x_general = torch.stack(
                    [normalize(image=x_)["image"] for x_ in x_base]
                ).cuda()
                y_pred = model(x_general)
                p_pred = y_pred.softmax(1)
                loss = loss_fn_cls(y_pred, y.cuda())
                total_loss_cls += loss.item()
                if args.method == "CR":
                    x_distorted = torch.stack(
                        [
                            normalize(image=x_)["image"]
                            for x_ in np.stack(
                                [distort_transform(image=x_)["image"] for x_ in x_base]
                            )
                        ]
                    ).cuda()
                    y_pred_distorted = model(x_distorted)
                    p_pred_distorted = y_pred_distorted.softmax(1)

                    # ---------------- CE ----------------
                    if args.cr_loss == "ce":
                        loss_cr = -(p_pred * y_pred_distorted.log_softmax(1)).sum(1).mean()

                    # ---------------- KL ----------------
                    elif args.cr_loss == "kl":
                        log_pd = torch.clamp(y_pred_distorted.log_softmax(1), -7, 0)
                        loss_cr = (p_pred * (p_pred.log() - log_pd)).sum(1).mean()

                    # ---------------- L2 ----------------
                    elif args.cr_loss == "l2":
                        loss_cr = ((p_pred - p_pred_distorted) ** 2).sum(1).mean()

                    # ---------------- JS ----------------
                    elif args.cr_loss == "js":
                        m = (p_pred + p_pred_distorted) / 2
                        loss_cr = (
                            F.kl_div(p_pred_distorted.log(), m, reduction="batchmean")
                            + F.kl_div(p_pred.log(), m, reduction="batchmean")
                        ) / 2
                    total_loss_cr += loss_cr.item()
                    loss = loss + args.lambda_ * loss_cr
            total_loss += loss.item()
            total_pred.append(y_pred.softmax(1).cpu().numpy())

    val_loss = total_loss / len(dataloader_val)
    val_loss_cls = total_loss_cls / len(dataloader_val)
    val_loss_cr = total_loss_cr / len(dataloader_val)
    total_pred = np.concatenate(total_pred)
    total_pred_hard = total_pred.argmax(1)
    val_acc = accuracy_score(dataloader_val.dataset.y, total_pred_hard)

    writer.add_scalar("Val/loss", val_loss, epoch)
    writer.add_scalar("Val/loss_cls", val_loss_cls, epoch)
    writer.add_scalar("Val/loss_cr", val_loss_cr, epoch)
    writer.add_scalar("Val/acc", val_acc, epoch)

    return val_loss_cls, total_pred


def inference(
    model,
    dataloader_test,
    transform,
    normalize,
    writer,
):
    model.eval()
    total_pred = []
    with torch.no_grad():
        for x, y in dataloader_test:
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                x_transformed = torch.stack(
                    [
                        normalize(image=x_)["image"]
                        for x_ in np.stack([transform(image=x_)["image"] for x_ in x])
                    ]
                ).cuda()
                y_pred = model(x_transformed)
            total_pred.append(y_pred.argmax(1).cpu().numpy())

    total_pred = np.concatenate(total_pred)
    test_acc = accuracy_score(dataloader_test.dataset.y, total_pred)

    writer.add_scalar("Test/acc", test_acc, 0)

    return total_pred, test_acc


class RandAugment(A.ImageOnlyTransform):
    def __init__(self, n, m, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.n = n
        self.m = m

        m_ratio = self.m / 30.0

        self.augment_list = (
            A.CLAHE(always_apply=True),
            A.Equalize(always_apply=True),
            A.InvertImg(always_apply=True),

            A.Rotate(limit=30 * m_ratio, always_apply=True),

            A.Posterize(num_bits=max(1, int(4 * m_ratio)), always_apply=True),

            A.Solarize(threshold=int(255 * m_ratio), always_apply=True),

            A.RGBShift(
                r_shift_limit=110 * m_ratio,
                g_shift_limit=110 * m_ratio,
                b_shift_limit=110 * m_ratio,
                always_apply=True,
            ),

            A.HueSaturationValue(
                hue_shift_limit=20 * m_ratio,
                sat_shift_limit=30 * m_ratio,
                val_shift_limit=20 * m_ratio,
                always_apply=True,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=m_ratio, contrast_limit=0, always_apply=True
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0, contrast_limit=m_ratio, always_apply=True
            ),

            A.ShiftScaleRotate(
                shift_limit=0.3 * m_ratio, shift_limit_y=0, rotate_limit=0, always_apply=True
            ),

            A.ShiftScaleRotate(
                shift_limit=0.3 * m_ratio, shift_limit_x=0, rotate_limit=0, always_apply=True
            ),

            A.CoarseDropout(max_holes=max(1, int(8 * m_ratio)), always_apply=True),

            A.Affine(shear=math.atan(0.3 * m_ratio), always_apply=True)
        )

        assert self.n <= len(self.augment_list)

    def apply(self, img, **params):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(image=img)["image"]
        return img


class DistortAugment(A.ImageOnlyTransform):
    def __init__(self, distort_level, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

        self.distort_level = distort_level - 1
        assert self.distort_level in [0, 1, 2, 3]

        distort_strength_list_blur = [0.5, 1, 1.5, 2]
        distort_strength_list_noise = [50, 100, 150, 200]
        distort_strength_list_brightness = [0.1, 0.2, 0.3, 0.4]
        distort_strength_list_radial = [0.25, 0.5, 0.75, 1]
        distort_strength_list_shear = [5, 10, 15, 20]

        strength_blur = distort_strength_list_blur[self.distort_level]
        strength_noise = distort_strength_list_noise[self.distort_level]
        strength_brightness = distort_strength_list_brightness[self.distort_level]
        strength_radial = distort_strength_list_radial[self.distort_level]
        strength_shear = distort_strength_list_shear[self.distort_level]

        # 내부에서 사용할 실제 Albumentations op들
        self.augment_list = [
            A.GaussianBlur(sigma_limit=(0, strength_blur), always_apply=True),
            A.GaussNoise(var_limit=(0, strength_noise), always_apply=True),
            A.RandomBrightnessContrast(
                brightness_limit=(-strength_brightness, strength_brightness),
                contrast_limit=0,
                always_apply=True,
            ),
            A.OpticalDistortion(
                distort_limit=(-strength_radial, strength_radial),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                always_apply=True,
            ),
            A.Affine(
                shear=(-strength_shear, strength_shear),
                interpolation=cv2.INTER_LINEAR,
                always_apply=True,
            )
        ]

    def apply(self, img, **params):
        # img만 받아서 변환 (Albumentations 약속방식)
        n = random.randint(1, 4)
        ops = random.sample(self.augment_list, k=n)
        for op in ops:
            img = op(image=img)['image']
        return img

import cv2
import albumentations as A

class DistortAugment_1(A.ImageOnlyTransform):
    def __init__(self, distort_type, distort_level, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

        self.distort_type = distort_type
        self.distort_level = distort_level - 1

        assert self.distort_level in [0, 1, 2, 3]

        distort_strength_list_blur = [0.5, 1, 1.5, 2]
        distort_strength_list_noise = [50, 100, 150, 200]
        distort_strength_list_brightness = [0.1, 0.2, 0.3, 0.4]
        distort_strength_list_radial = [0.25, 0.5, 0.75, 1]
        distort_strength_list_shear = [5, 10, 15, 20]

        strength_blur = distort_strength_list_blur[self.distort_level]
        strength_noise = distort_strength_list_noise[self.distort_level]
        strength_brightness = distort_strength_list_brightness[self.distort_level]
        strength_radial = distort_strength_list_radial[self.distort_level]
        strength_shear = distort_strength_list_shear[self.distort_level]

        # --- 선택된 distort_type에 맞는 albumentation augmentation 준비 ---
        if self.distort_type == "blur":
            self.augment = A.GaussianBlur(
                sigma_limit=(0, strength_blur),
                p=1.0
            )

        elif self.distort_type == "noise":
            self.augment = A.GaussNoise(
                var_limit=(0, strength_noise),
                p=1.0
            )

        elif self.distort_type == "brightness":
            self.augment = A.RandomBrightnessContrast(
                brightness_limit=(-strength_brightness, strength_brightness),
                contrast_limit=0,
                p=1.0
            )

        elif self.distort_type == "radial":
            self.augment = A.OpticalDistortion(
                distort_limit=(-strength_radial, strength_radial),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            )

        elif self.distort_type == "shear":
            self.augment = A.Affine(
                shear=(-strength_shear, strength_shear),
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            )

        else:
            raise ValueError(f"Unknown distort_type: {self.distort_type}")

    def apply(self, img, **params):
        # Albumentations의 transform은 항상 {"image": img} 형태 반환 → 이 형식 맞춰 적용
        return self.augment(image=img)["image"]
