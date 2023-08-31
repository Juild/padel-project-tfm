
import pickle
import sys

import numpy as np
from datapoint_class import ImageDP
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from utilities.preprocessing import Mask

class ImageDataset(Dataset):
    def __init__(self, images: List[ImageDP], transforms=None, target_transforms=None) -> None:
        self.transforms = transforms
        self.target_transforms = target_transforms
        print(type(images[0]))
        # assert isinstance(images[0], ImageDP), f'Images provided in the list are not [ImageDP] instances'
        self.images = images
        self.means = None
        self.stds = None
    def __getitem__(self, idx):
        image = self.images[idx]
        label = image.label


        # Normalize always
        image_tensor = image.to_tensor()
 
        #  All Pytorch models need their input to be "channel first"
        # i.e. TensorSize(C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        if self.transforms:
            image_tensor = self.transforms(image_tensor)
        image_tensor = 2*(image_tensor/255.) - 1
        normalize = Normalize(mean=self.means, std=self.stds)
        image_tensor = normalize(image_tensor)

        return image_tensor, label

    def __len__(self):
        # return the size of the first index in the stack of image tensors, i.e. the number of images tensors
        # that is, the number of images in the dataset
        return len(self.images)

    def apply_mask(self, mask: Mask):
        for image in self.images:
            # assert np.min(image.arr) >= 0, 'Mask is only applied to an image with channels in range [0, 255]'
            image.apply_transform([mask])

    def get_channel_mean_std(self):
        blue_channel = []
        green_channel = []
        red_channel = []
        h, w = self.images[0].arr.shape[:2]
        for image in self.images:
            image_arr = image.arr
            image_arr = 2 * (image_arr/255.) - 1
            blue_channel.append(image_arr[:, :, 0].reshape(h * w))
            green_channel.append(image_arr[:, :, 1].reshape(h * w))
            red_channel.append(image_arr[:, :, 2].reshape(h * w))
        blue_channel = Tensor(np.array(blue_channel))
        green_channel = Tensor(np.array(green_channel))
        red_channel = Tensor(np.array(red_channel))

        self.means = [blue_channel.mean().item(), green_channel.mean().item(),
                red_channel.mean().item()]
        self.stds = [blue_channel.std().item(), green_channel.std().item(),
            red_channel.std().item()]
        print('MEANS', self.means, 'STDS', self.stds)
        with open('mean.dat', 'wb') as f:
            pickle.dump(self.means, f)
        with open('stds.dat', 'wb') as f:
            pickle.dump(self.stds, f)

                

