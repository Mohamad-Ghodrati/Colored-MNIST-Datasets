"""
This module contains a class for creating and managing `10CMNIST` dataset.
"""

import torch

from torch.utils.data import Dataset
from torchvision import datasets
from utils import COLORS


class CMNIST10(Dataset):
    """
    Custom Dataset class to create 10CMNIST.
    Each digit class is assigned a unique color for the training set.
    In the test set, digits are colored with a random color that is not their associated label's color.

    Args:
        root (str): Directory path to save or load the MNIST data.
        download (bool, optional): If True, download MNIST if not present. Default is True.
        colors (tuple[tuples], optional): A tuples of RGB color values corresponding to each digit class (0-9).
        train (bool, optional): Use the training or test split of MNIST. Default is True.
        transform (torch.nn.Module, optional): Optional transformations for MNIST images. Default is None.
        downsample (bool, optional): If True, downsample MNIST images from 28x28 to
                                     14x14. Default is True.

    Caution:
        - Note that only the red and green channels are used for colorization in this implementation.
        - The blue channel is not utilized.
        - If desired, an additional channel can be added to the output for visualization,
          to make it a standard 3-channel RGB image.


    """

    def __init__(
        self,
        root: str,
        download: bool = True,
        colors: tuple[tuple] = COLORS,
        train: bool = True,
        transform: torch.nn.Module | None = None,
        downsample: bool = True,
    ):

        self.images, self.labels = self._load_mnist(root, train, download, transform)
        self.colors = colors
        self.train = train
        self.downsample = downsample
        self.transform = transform

    @staticmethod
    def _load_mnist(root, train, download, transform):
        mnist = datasets.MNIST(
            root, train=train, download=download, transform=transform
        )
        return mnist.data, mnist.targets

    def _change_color(self, image, label):
        if self.train:
            color = self.colors[label]
        else:
            colors = self.colors[:label] + self.colors[label + 1 :]
            color = colors[torch.randint(0, 9, size=(1,))]
        r = ((image / 255) * color[0]) / 255
        g = ((image / 255) * color[1]) / 255
        colored_image = torch.stack([r, g])
        return colored_image

    def __getitem__(self, index):
        image = self.images[index][::2, ::2] if self.downsample else self.images[index]
        label = self.labels[index]
        image = self._change_color(image, label)
        if self.transform is not None:
            return self.transform(image), label
        return image, label

    def __len__(self):
        return len(self.labels)
