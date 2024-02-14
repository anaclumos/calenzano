import torch
import torch.utils.data
from torchvision import transforms
import numpy as np

class GridTransform:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        blank = torch.zeros_like(tensor)
        slice_height = tensor.shape[1] // 2
        slice_width = tensor.shape[2] // 2

        slices = (
            slice(None),
            slice(slice_height * self.row, slice_height * (self.row + 1)),
            slice(slice_width * self.col, slice_width * (self.col + 1)),
        )
        blank[slices] = tensor[slices]
        return (blank - 0.5) / 0.5

class ScatterTransform:
    def __init__(self, seed, idx, desired_dataset_count):
        self.seed = seed
        self.idx = idx
        self.desired_dataset_count = desired_dataset_count

    def __call__(self, img):
        # Ensure input is a tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        # Set RNG seed for consistent randomness
        np.random.seed(self.seed)

        # Initialize an empty tensor of the same shape as img but filled with zeros
        output = torch.zeros_like(img)

        # Iterate over each pixel and decide which dataset it belongs to
        for i in range(img.shape[1]):  # Height
            for j in range(img.shape[2]):  # Width
                # Generate a random dataset index for the current pixel
                dataset_idx = np.random.randint(0, self.desired_dataset_count)
                
                # If the random index matches this instance's idx, copy the pixel
                if dataset_idx == self.idx:
                    output[:, i, j] = img[:, i, j]

        return output

class ScatterAllowingDupedTransform:
    def __init__(self, idx, desired_dataset_count):
        self.idx = idx
        self.desired_dataset_count = desired_dataset_count
        self.inclusion_probability = 1 / self.desired_dataset_count

    def __call__(self, img):
        # Ensure input is a tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        # Initialize an empty tensor of the same shape as img but filled with zeros
        output = torch.zeros_like(img)

        # Iterate over each pixel and decide if it belongs to this dataset based on probability
        for i in range(img.shape[1]):  # Height
            for j in range(img.shape[2]):  # Width
                # Determine if the pixel should be included in this dataset
                if np.random.rand() < self.inclusion_probability:
                    output[:, i, j] = img[:, i, j]

        return output
