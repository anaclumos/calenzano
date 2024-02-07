import torch
import torch.utils.data
from torchvision import transforms


class GridTransform:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        blank = torch.zeros_like(tensor)
        slice_height = tensor.shape[1] // 4
        slice_width = tensor.shape[2] // 4

        slices = (
            slice(None),
            slice(slice_height * self.row, slice_height * (self.row + 1)),
            slice(slice_width * self.col, slice_width * (self.col + 1)),
        )
        blank[slices] = tensor[slices]
        return (blank - 0.5) / 0.5
