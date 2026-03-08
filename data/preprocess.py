import numpy as np
import pickle
from torchvision import transforms, datasets
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os


def cifar10():
    dataset_train = datasets.CIFAR10(root='./', train=True, download=True, transform=None)
    dataset_test = datasets.CIFAR10(root='./', train=False, download=True, transform=None)

    X_train = dataset_train.data
    y_train = dataset_train.targets
    X_test = dataset_test.data
    y_test = dataset_test.targets

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    print(X.shape, y.shape, X.dtype, y.dtype)

    np.save('processed/CIFAR10_X.npy', X)
    np.save('processed/CIFAR10_y.npy', y)

    mean = np.mean(X_train / 255, axis=(0, 1, 2))
    std = np.std(X_train / 255, axis=(0, 1, 2))

    np.save('processed/CIFAR10_mean.npy', mean)
    np.save('processed/CIFAR10_std.npy', std)

def cifar100():
    dataset_train = datasets.CIFAR100(root='./', train=True, download=True, transform=None)
    dataset_test = datasets.CIFAR100(root='./', train=False, download=True, transform=None)

    X_train = dataset_train.data
    y_train = dataset_train.targets
    X_test = dataset_test.data
    y_test = dataset_test.targets

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    print(X.shape, y.shape, X.dtype, y.dtype)

    np.save('processed/CIFAR100_X.npy', X)
    np.save('processed/CIFAR100_y.npy', y)

    mean = np.mean(X_train / 255, axis=(0, 1, 2))
    std = np.std(X_train / 255, axis=(0, 1, 2))

    np.save('processed/CIFAR100_mean.npy', mean)
    np.save('processed/CIFAR100_std.npy', std)

def svhn():
    # SVHN
    dataset_train = datasets.SVHN(root='./', split='train', download=True, transform=None)
    dataset_test = datasets.SVHN(root='./', split='test', download=True, transform=None)

    X_train = dataset_train.data
    y_train = dataset_train.labels
    X_test = dataset_test.data
    y_test = dataset_test.labels

    X = np.concatenate((X_train, X_test))
    X = np.transpose(X, (0, 2, 3, 1))
    y = np.concatenate((y_train, y_test))

    print(X.shape, y.shape, X.dtype, y.dtype)

    np.save('processed/SVHN_X.npy', X)
    np.save('processed/SVHN_y.npy', y)

def remove_boundaries(img):
    percentile = np.percentile(img[:,:,0], 10)
    is_img_w = np.where(img[np.linspace(0, img.shape[0], 20, dtype=np.int32)[7:12],:,0].min(0)> np.min([percentile, 30]))
    is_img_h = np.where(img[:, np.linspace(0, img.shape[1], 20, dtype=np.int32)[7:12],0].min(1)> np.min([percentile, 30]))
    img = img[is_img_h[0][0]:is_img_h[0][-1], is_img_w[0][0]:is_img_w[0][-1]]
    return img

def kvasir(resize_size = 128):
    class_list = os.listdir('kvasir-dataset-v2')
    file_list = os.listdir('kvasir-dataset-v2/' + class_list[1])
    filename = 'kvasir-dataset-v2/' + class_list[1] + '/' + file_list[0]
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_list = []
    label_list = []
    for i, class_name in enumerate(class_list):
        print(i, class_name)
        file_list = os.listdir('kvasir-dataset-v2/' + class_list[i])
        for j in range(len(file_list)):
            filename = 'kvasir-dataset-v2/' + class_list[i] + '/' + file_list[j]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = remove_boundaries(img)
            img = cv2.resize(img, (resize_size, resize_size))
            img_list.append(img)
            label_list.append(i)

    X = np.array(img_list)
    y = np.array(label_list)
    np.save('processed/kvasir_X.npy', X)
    np.save('processed/kvasir_y.npy', y)


def mvtec(resize_size=128):
    class_list = sorted([f for f in os.listdir('mvtec') if '.' not in f])
    image_list = []
    label_list = []
    original_shape_list = []
    for i, class_name in enumerate(class_list):
        print(i, class_name)
        file_list = os.listdir('mvtec/' + class_list[i] + '/train/good/')
        for j in range(len(file_list)):
            filename = 'mvtec/' + class_list[i] + '/train/good/' + file_list[j]
            img = cv2.imread(filename)
            original_shape_list.append(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize_size, resize_size))
            image_list.append(img)
            label_list.append(i)

        type_list = os.listdir('mvtec/' + class_list[i] + '/test')
        for type_name in type_list:
            file_list = os.listdir('mvtec/' + class_list[i] + '/test/' + type_name)

            if type_name == 'good':
                for k in range(len(file_list)):
                    filename = 'mvtec/' + class_list[i] + '/test/' + type_name + '/' + file_list[k]
                    img = cv2.imread(filename)
                    original_shape_list.append(img.shape)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (resize_size, resize_size))
                    image_list.append(img)
                    label_list.append(i)
            else:
                for k in range(len(file_list)):
                    filename = 'mvtec/' + class_list[i] + '/test/' + type_name + '/' + file_list[k]
                    img = cv2.imread(filename)
                    original_shape_list.append(img.shape)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (resize_size, resize_size))
                    image_list.append(img)
                    label_list.append(i+15)
    y = np.array(label_list)
    X = np.array(image_list)

    np.save('processed/mvtec_X.npy', X)
    np.save('processed/mvtec_y.npy', y)


def ham10000(resize_size=128):
    file_list = sorted(os.listdir('HAM10000'))
    file_list = [f for f in file_list if f.endswith('.jpg')]
    mapping_type = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    df = pd.read_csv('HAM10000/HAM10000_metadata').sort_values('image_id')
    img_list = []
    for filename in file_list:
        img = cv2.imread('HAM10000/' + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (resize_size, resize_size))
        img_list.append(img)

    X = np.array(img_list)
    y = df['dx'].map(mapping_type).values
    np.save('processed/HAM10000_X.npy', X)
    np.save('processed/HAM10000_y.npy', y)


def cifar100C():
    c_root = "CIFAR-100-C"   # CIFAR-10-C 압축 푼 폴더
    save_root = "processed"

    os.makedirs(save_root, exist_ok=True)

    # CIFAR-10-C labels (모든 corruption 공통)
    labels = np.load(os.path.join(c_root, "labels.npy")).astype(np.int64)

    # CIFAR-10-C corruption 종류 리스트
    corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]

    print("Found corruptions:", corruptions)

    # 각 corruption별로 저장
    for cor in corruptions:
        file_path = os.path.join(c_root, f"{cor}.npy")
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        images = np.load(file_path).astype(np.uint8)  # (10000, 32, 32, 3)
        print(f"{cor}: {images.shape}, dtype={images.dtype}")

        np.save(os.path.join(save_root, f"CIFAR100C_{cor}_X.npy"), images)
    np.save(os.path.join(save_root, f"CIFAR100C_y.npy"), labels)

    print("Done!")

def cifar100C():
    c_root = "CIFAR-10-C"   # CIFAR-10-C 압축 푼 폴더
    save_root = "processed"

    os.makedirs(save_root, exist_ok=True)

    # CIFAR-10-C labels (모든 corruption 공통)
    labels = np.load(os.path.join(c_root, "labels.npy")).astype(np.int64)

    # CIFAR-10-C corruption 종류 리스트
    corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]

    print("Found corruptions:", corruptions)

    # 각 corruption별로 저장
    for cor in corruptions:
        file_path = os.path.join(c_root, f"{cor}.npy")
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        images = np.load(file_path).astype(np.uint8)  # (10000, 32, 32, 3)
        print(f"{cor}: {images.shape}, dtype={images.dtype}")

        np.save(os.path.join(save_root, f"CIFAR10C_{cor}_X.npy"), images)
    np.save(os.path.join(save_root, f"CIFAR10C_y.npy"), labels)

    print("Done!")