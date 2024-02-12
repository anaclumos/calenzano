import threading

import torch.utils.data
from torchvision import datasets, transforms


class DataProvider:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, transform=None):
        """Initialize the provider with a specific or default transform for data preprocessing."""
        if transform is None:
            self.transform = self.default_transform()
        else:
            self.transform = transform
        self._initialize_data()

    @classmethod
    def get_instance(cls, key, transform=None):
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(transform)
        return cls._instances[key]

    @staticmethod
    def default_transform():
        """Define the default transform to be used if no transform is provided."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _initialize_data(self):
        self.trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=64, shuffle=True
        )

        self.testset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=64, shuffle=False
        )

    def get_loaders(self):
        return self.trainloader, self.testloader
