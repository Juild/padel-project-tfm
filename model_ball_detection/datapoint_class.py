import torch
import cv2
import numpy as np
from torch import Tensor


class ImageDP:
    def __init__(self, arr=None, label=None) -> None:
        self._arr = arr
        self._label: int = label

    @classmethod
    def from_file(cls, dir: str):
        image_arr = cv2.imread(dir)
        return cls(arr=image_arr)
        
    @classmethod
    def from_array(cls, arr: np.array):
        return cls(arr=arr)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: int):
        self._label = label

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, arr):
        self._arr = arr

    def visualize(self):
        cv2.imshow('ImageDP', self._arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, id: int):
        cv2.imwrite(f'./input_training_data/input{id}.jpg', self._arr)

    def apply_transform(self, transforms: list):
        for transform in transforms:
            self._arr = transform(self._arr)

    def to_tensor(self):
        return torch.tensor(self._arr, dtype=torch.uint8)
