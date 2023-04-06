import torch
import torch.utils.data
import cv2
import os
import numpy as np
from torchvision import datasets
from util.util import plot_images
from util.img_transform import ImgTransform
from dataset.sampler import valid_and_train_samplers
from dataset.dataset import TestImageFolder

def get_train_valid_loader(data_dir: str, 
                            batch_size: int, 
                            data_transforms: dict, 
                            random_state: int, 
                            weighted_sampler: bool,
                            valid_size: float,
                            shuffle: bool,
                            show_sample: bool,
                            num_workers: int,
                            pin_memory: bool):

    # load the dataset
    train_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    valid_dataset = datasets.ImageFolder(data_dir, data_transforms['valid'])

    train_sampler, valid_sampler, cls_to_weight = valid_and_train_samplers(
        train_dataset, weighted_sampler, valid_size, random_state)

    train_dataset.cls_to_weight = cls_to_weight

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, 
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, 
        pin_memory=pin_memory,
    )

    print(f'Total: {len(train_dataset)}; Train/Valid: {len(train_sampler)}/{len(valid_sampler)}')

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=5*8, sampler=train_sampler, num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, data_dir, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir: str,
                    batch_size: int,
                    data_transforms: dict,
                    num_workers: int,
                    pin_memory: bool):

    test_dataset = TestImageFolder(data_dir, data_transforms['valid'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader

def train_loader_handcrafted(img_paths):
    data = []
    labels = []

    for(i, img_path) in enumerate(img_paths):
        img = cv2.imread(img_path)
        label = img_path.split(os.path.sep)[-2]
        # preprocess image
        data.append(img)
        labels.append(label)
    
    return (np.array(data), np.array(labels))
