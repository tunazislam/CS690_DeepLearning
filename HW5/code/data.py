import numpy as np

import torch
import torchvision


class ColorShift:
    """Shift the Hue of the images"""

    def __init__(self, amount):
        assert -0.5 <= amount <= 0.5
        self.amount = amount

    def __call__(self, x):
        return torchvision.transforms.functional.adjust_hue(x, self.amount)


def prepare_data(data_folder, batch_size):

    # Load / Download the dataset and build the shifted version

    mean = [0.491, 0.482, 0.447]
    std = [0.247, 0.243, 0.262]
    shift_amount = 0.25
    cifar10_train = torchvision.datasets.CIFAR10(
        root=data_folder,
        train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]),
        download=True)

    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_folder,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]),
        download=True)

    cifar10_shifted_train = torchvision.datasets.CIFAR10(
        root=data_folder,
        train=True,
        transform=torchvision.transforms.Compose([
            ColorShift(shift_amount),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]),
        download=False)

    cifar10_shifted_test = torchvision.datasets.CIFAR10(
        root=data_folder,
        train=False,
        transform=torchvision.transforms.Compose([
            ColorShift(shift_amount),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]),
        download=False)

    # Split the datasets into train, validation and test

    indices = np.random.permutation(len(cifar10_train))
    split_point = (len(indices) // 10) * 9
    train_indices = indices[:split_point]
    validation_indices = indices[split_point:]

    source_train_dataset = torch.utils.data.Subset(cifar10_train, train_indices)
    source_validation_dataset = torch.utils.data.Subset(cifar10_train, validation_indices)
    source_test_dataset = cifar10_test

    target_train_dataset = torch.utils.data.Subset(cifar10_shifted_train, train_indices)
    target_validation_dataset = torch.utils.data.Subset(cifar10_shifted_train, validation_indices)
    target_test_dataset = cifar10_shifted_test

    train_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True, num_workers=4)
    test_args = dict(batch_size=1024, pin_memory=True, drop_last=False, shuffle=False, num_workers=4)

    # Build the data loaders

    source_train_loader = torch.utils.data.DataLoader(source_train_dataset, **train_args)
    source_validation_loader = torch.utils.data.DataLoader(source_validation_dataset, **test_args)
    source_test_loader = torch.utils.data.DataLoader(source_test_dataset, **test_args)

    target_train_loader = torch.utils.data.DataLoader(target_train_dataset, **train_args)
    target_validation_loader = torch.utils.data.DataLoader(target_validation_dataset, **test_args)
    target_test_loader = torch.utils.data.DataLoader(target_test_dataset, **test_args)

    return {
        'source': {
            'train': source_train_loader,
            'validation': source_validation_loader,
            'test': source_test_loader,
        },
        'target': {
            'train': target_train_loader,
            'validation': target_validation_loader,
            'test': target_test_loader,
        },
    }
