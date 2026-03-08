import numpy as np
import pickle
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import models
import cv2
from tqdm import tqdm


from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

# summary writer
from torch.utils.tensorboard import SummaryWriter
from wideresnet import WideResNet


from utils import *


start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="CR", choices=["CR", "general", "distort", "combined", "finetune", "augmix"])
parser.add_argument("--lambda_", type=float, default=1.0)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100", "svhn", "HAM10000", "mvtec", "kvasir"],
)
parser.add_argument("--model_architecture", type=str, default="vgg19_bn")
parser.add_argument("--max_epoch", type=int, default=1)
parser.add_argument("--exp_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-8)
parser.add_argument("--cr_loss", type=str, default="kl", choices=["kl", "l2", "ce", "js"])
parser.add_argument("--distortion_level", type=int, default=1, choices=[0, 1, 2, 3, 4])
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--num_views", type=int, default=3)
parser.add_argument("--date", type=str, default="0220")

args = parser.parse_args()


start_time = time.time()
print(args)
# reduce batch size if big dataset
if args.dataset in ["HAM10000", "mvtec", "kvasir"]:
    args.batch_size = args.batch_size // 4
    dataset_name = args.dataset
else:
    dataset_name = args.dataset.upper()

# Load data
filename = os.path.expanduser(f"../../data/processed/{dataset_name}")

X = np.load(filename+'_X.npy', allow_pickle=True)
Y = np.load(filename+'_y.npy', allow_pickle=True)

num_classes = Y.max().item() + 1

save_dir = f"{args.exp_id}/{args.dataset}/{args.model_architecture}/{args.method}/{args.cr_loss}/{args.lambda_}/{args.distortion_level}/{args.num_views}"

os.makedirs(f"{args.date}/log/{save_dir}", exist_ok=True)
os.makedirs(f"{args.date}/model/{save_dir}", exist_ok=True)
os.makedirs(f"{args.date}/result/{save_dir}", exist_ok=True)

# Split data
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=args.exp_id, stratify=Y
)
# X_trainval, X_test = X[:50000], X[50000:]
# y_trainval, y_test = Y[:50000], Y[50000:]


X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=1 / 9,
    random_state=args.exp_id,
    stratify=y_trainval,
)

if args.max_epoch == 1:
    X_train = X_train[:1000]
    y_train = y_train[:1000]

mean = np.mean(X_train / 255, axis=(0, 1, 2))
std = np.std(X_train / 255, axis=(0, 1, 2))

# Dataset
dataset_train = CustomDataset(X_train, y_train)
dataset_val = CustomDataset(X_val, y_val)
dataset_test = CustomDataset(X_test, y_test)

# Dataloader
dataloader_train = DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

if args.dataset in ["cifar10", "cifar100", "svhn"]:
    general_transform = A.Compose(
        [
            RandAugment(n=2, m=9),
            A.HorizontalFlip(p=0.5),
        ]
    )

elif args.dataset in ["svhn"]:
    general_transform = A.Compose(
        [
            RandAugment(n=2, m=9),
        ]
    )

elif args.dataset in ["HAM10000", "mvtec", "kvasir"]:
    general_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.1),
                translate_percent=(-0.1, 0.1),
                shear=(-11, 11),
                rotate=(-30, 30),
            ),
        ]
    )  # , iaa.Multiply((0.8, 1.2)), iaa.ContrastNormalization((0.8, 1.2))])

normalize = A.Compose([A.Normalize(mean=mean.tolist(), std=std.tolist()), ToTensorV2()])

if args.distortion_level == 0:
    distort_transform = A.Compose([])
elif args.distortion_level > 0:
    distort_transform = DistortAugment(distort_level=args.distortion_level)

if args.method in ["distort", "finetune"]:
    general_transform = distort_transform
elif args.method == "combined":
    general_transform = A.OneOf([general_transform, distort_transform])

if args.model_architecture == "vgg19_bn":
    model = models.vgg19_bn(weights=None)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.classifier = torch.nn.Linear(512, num_classes)

elif args.model_architecture == "wide_resnet50_2":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', weights=None)
    model.fc = torch.nn.Linear(2048, num_classes)

elif args.model_architecture == "resnet50":
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, num_classes)

elif args.model_architecture == "wrn40-4": # wrn40-4
    model = WideResNet(40, num_classes, 4)

model.cuda()

if args.method == "finetune":
    save_dir_general = f"{args.exp_id}/{args.dataset}/{args.model_architecture}/general/{args.cr_loss}/{args.lambda_}/1/{args.num_views}"
    model.load_state_dict(torch.load(f"{args.date}/model/{save_dir_general}/best_model.pt", weights_only=True))
    args.lr = 1e-4

# optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)

loss_fn_cls = torch.nn.CrossEntropyLoss()


writer = SummaryWriter(f"{args.date}/log/{save_dir}")

val_log = []



for epoch in tqdm(range(args.max_epoch), desc="Training Epochs"):
    train_loss_cls = train(
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
    )
    val_loss_cls, total_pred = validate(
        model,
        dataloader_val,
        loss_fn_cls,
        general_transform,
        distort_transform,
        normalize,
        epoch,
        writer,
        args,
    )
    val_log.append(val_loss_cls)
    val_acc = np.mean(total_pred.argmax(1) == y_val)
    if epoch == np.argmin(val_log):
        torch.save(model.state_dict(), f"{args.date}/model/{save_dir}/best_model.pt")

    elif epoch - np.argmin(val_log) > args.patience:
        break

    lr_scheduler.step(val_loss_cls)
    writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

    # stop if nan
    if np.isnan(val_loss_cls):
        print("nan")
        break

# load best model
model.load_state_dict(torch.load(f"{args.date}/model/{save_dir}/best_model.pt", weights_only=True))

test_acc_list = []

# clean test
no_aug = A.Compose([])
total_pred, test_acc = inference(model, dataloader_test, no_aug, normalize, writer)
print(f"Test Acc {test_acc:.4f} | Time: {(time.time() - start)/60:.2f} min")
test_acc_list.append(test_acc)

# distortion test

aug_list = ["blur", "noise", "brightness", "radial", 'shear']
# aug_list = ["noise"]

for aug in aug_list:
    if aug == "blur":
        distortion_level_list = [0.5, 1, 1.5, 2]
    elif aug == "noise":
        distortion_level_list = [50, 100, 150, 200]
    elif aug == "brightness":
        distortion_level_list = [0.1, 0.2, 0.3, 0.4]
    elif aug == "radial":
        distortion_level_list = [0.25, 0.5, 0.75, 1.0]
    elif aug == "shear":
        distortion_level_list = [5, 10, 15, 20]

    for distortion_level in distortion_level_list:
        if aug == "blur":
            distort_transform = A.GaussianBlur(
                sigma_limit=(distortion_level, distortion_level), p=1
            )
        elif aug == "noise":
            distort_transform = A.GaussNoise(
                var_limit=(distortion_level, distortion_level), p=1
            )
        elif aug == "brightness":
            distort_transform = A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(distortion_level, distortion_level),
                        p=1,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-distortion_level, -distortion_level),
                        p=1,
                    ),
                ]
            )

        elif aug == "radial":
            distort_transform = A.OneOf(
                [
                    A.OpticalDistortion(
                        distort_limit=(distortion_level, distortion_level), p=1
                    ),
                    A.OpticalDistortion(
                        distort_limit=(-distortion_level, -distortion_level), p=1
                    ),
                ]
            )

        elif aug == "shear":
            distort_transform = A.Affine(
                        shear=(distortion_level, distortion_level), p=1
                    )
        total_pred, test_acc = inference(
            model, dataloader_test, distort_transform, normalize, writer
        )
        test_acc_list.append(test_acc)

    for i, test_acc in enumerate(test_acc_list):
        writer.add_scalar(f"Test/{aug}", test_acc, i)

    np.save(f"{args.date}/result/{save_dir}/{aug}_test_acc.npy", test_acc_list)

    # delete last three values of list
    test_acc_list = test_acc_list[:-4]
writer.close()
