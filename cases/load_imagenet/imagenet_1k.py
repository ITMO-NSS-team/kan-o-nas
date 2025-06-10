import os
import json
import torch
import numpy as np
import torchvision
from PIL import Image
from datasets import load_dataset
# from torchinfo import summary
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy

from cases.load_imagenet.classification_dataset import Classification
from cases.load_imagenet.focal_loss import FocalLoss


# from models import reskagnet50
# from train import Classification, train_model, FocalLoss
# from utils import GradCAMReporter


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def unpack_and_save_chunk(dataset, path2chunk):
    os.makedirs(path2chunk, exist_ok=True)
    train_data = []
    for i in range(len(dataset)):
        sample = dataset[i]
        sample, label = sample['image'].convert('RGB'), sample['label']
        sample.save(os.path.join(path2chunk, f'img_{i}.jpg'))
        train_data.append({'image': os.path.join(path2chunk, f'img_{i}.jpg'), 'label': label})
    return train_data


def check_and_load_chunck(dataset, cache_dir, pack_name):
    if os.path.exists(os.path.join(cache_dir, '.'.join([pack_name, 'json']))):
        with open(os.path.join(cache_dir, '.'.join([pack_name, 'json'])), 'r') as f:
            _data = json.load(f)
    else:
        _data = unpack_and_save_chunk(dataset, os.path.join(cache_dir, pack_name))
        with open(os.path.join(cache_dir, '.'.join([pack_name, 'json'])), 'w') as f:
            json.dump(_data, f)
    return _data


def unpack_imagenet(dataset, cache_dir='./data/imagenet1k_unpacked'):
    print('READ TRAIN')
    train_data = check_and_load_chunck(dataset['train'], cache_dir, 'train')
    print('READ VALIDATION')
    validation_data = check_and_load_chunck(dataset['validation'], cache_dir, 'validation')
    print('READ TEST')
    test_data = check_and_load_chunck(dataset['test'], cache_dir, 'test')
    return train_data, validation_data, test_data


def get_imagenet():
    dataset = load_dataset("imagenet-1k", cache_dir='~/data/imagenet1k', trust_remote_code=True)

    # if cfg.unpack_data:
    #     train_data, validation_data, test_data = unpack_imagenet(dataset, cache_dir='./data/imagenet1k_unpacked')
    #     del dataset
    # else:
    train_data, validation_data, test_data = dataset['train'], dataset['validation'], dataset['test']

    transforms_train = v2.Compose([
        v2.ToImagePIL() if int(torchvision.__version__.split(".")[1]) <= 15 else v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(224, antialias=True),
        v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                         v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                         # v2.AutoAugment(AutoAugmentPolicy.SVHN),
                         # v2.TrivialAugmentWide()
                         ]),
        v2.ToTensor(),
        v2.ToDtype(torch.float32),
        v2.Lambda(lambda x: x / 255.0),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_val = v2.Compose([
        v2.ToImagePIL() if int(torchvision.__version__.split(".")[1]) <= 15 else v2.ToImage(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.ToDtype(torch.float32),
        v2.Lambda(lambda x: x / 255.0),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def one_hot_encode(target):
        return torch.nn.functional.one_hot(torch.tensor(target), num_classes=1000).float()

    train_dataset = Classification(train_data, transform=transforms_train, target_transform=one_hot_encode)
    # train_dataset = Classification(dataset['validation'], transform=transforms_train, target_transform=one_hot_encode)
    val_dataset = Classification(validation_data, transform=transforms_val, target_transform=one_hot_encode)
    # test_dataset = Classification(test_data, transform=transforms_val, target_transform=one_hot_encode)

    # return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset


def main():
    dataset_train, dataset_val = get_imagenet()
    # loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
    loss_func = FocalLoss(gamma=1.5)

    print(len(dataset_train))
    print(dataset_train[0][0].size())

    # train_model(model, dataset_train, dataset_val, loss_func, cfg, dataset_test=None, cam_reporter=None)
    # eval_model(model, dataset_test, cfg)


if __name__ == '__main__':
    main()
