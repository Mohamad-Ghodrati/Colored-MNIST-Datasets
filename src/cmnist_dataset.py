"""
This module contains a class for creating and managing `CMNIST` dataset.
"""

import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision import datasets
from torchvision.utils import make_grid


class CMNISTDataset(Dataset):
    """
    A dataset class to handle Colored MNIST (CMNIST) samples.

    Each sample includes:
    - A transformed or raw grayscale image represented as a tensor.
    - A binary label indicating whether the digit belongs to the range [0-4] (label 0) or [5-9] (label 1).
    - An environment identifier, which tracks the environment this sample belongs to.

    Args:
        images (torch.Tensor): Input image tensors of shape `(N, H, W)`.
        labels (torch.Tensor): Binary labels of shape `(N,)`.
        environment (int): Identifier for the dataset's environment.
        transform (torch.nn.Module, optional): Transformations to apply to the images.
                                                Defaults to None.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        environment: torch.Tensor | int,
        transform: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.images = images
        self.labels = labels
        self.environment = environment
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return (
                self.transform(self.images[index]),
                self.labels[index],
                self.environment,
            )
        return self.images[index], self.labels[index], self.environment

    def __len__(self):
        return len(self.labels)


class CMNIST:
    """
    A class to create and manage Colored MNIST datasets with environmental factors.

    This class partitions the MNIST dataset into subsets for different environments,
    with user-defined probabilities of flipping (label-based) image colors.

    Args:
        root (str): Directory path to save or load the MNIST data.
        download (bool, optional): If True, download MNIST if not present. Default is True.
        sizes (tuple[int] | tuple[float], optional): Sizes of subsets for each environment.
            - If integers, each value specifies the number of samples in an environment.
            - If floats, they represent percentages of the dataset and must sum to 1.
            Default is (0.4, 0.4, 0.2).
        e_s (tuple[float], optional): Environment probabilities for flipping colors. Default is (0.1, 0.3, 0.9).
        train (bool, optional): Use the training or test split of MNIST. Default is True.
        shuffle (bool, optional): Shuffle the dataset before partitioning. Default is True.
        p_flip_label (float, optional): Probability of flipping binary labels. Default is 0.25.
        transform (torch.nn.Module, optional): Optional transformations for MNIST images. Default is None.
        downsample (bool, optional): If True, downsample MNIST images from 28x28 to 14x14. Default is True.

    Caution:
    - Note that only the red and green channels are used for colorization in this implementation.
    - The blue channel is not utilized.
    - If desired, an additional channel can be added to the output for visualization,
        to make it a standard 3-channel RGB image.

    Raises:
        ValueError: If `sizes` are floats and do not sum to 1.
        ValueError: If `sizes` are ints and sum of them exceeds available samples.
        ValueError: If `sizes` and `e_s` lengths do not match.
    """

    def __init__(
        self,
        root: str,
        download: bool = True,
        sizes: tuple[int] | tuple[float] = (0.4, 0.4, 0.2),
        e_s: tuple[float] = (0.1, 0.3, 0.9),
        train: bool = True,
        shuffle: bool = True,
        p_flip_label: float = 0.25,
        transform: torch.nn.Module | None = None,
        downsample: bool = True,
    ):
        self.mnist = self._load_mnist(root, train, download, transform)
        self.sizes = sizes
        self.e_s = e_s
        self.train = train
        self.shuffle = shuffle
        self.p_flip_label = p_flip_label
        self.downsample = downsample

        if len(self.sizes) != len(self.e_s):
            raise ValueError("The number of sizes and e_s must match.")
        if isinstance(self.sizes[0], float):
            if sum(self.sizes) > 1.0:
                raise ValueError("If sizes are floats, their sum cannot exceed 1.0.")
            self.sizes = [int(size * len(self.mnist)) for size in self.sizes]
        else:
            if sum(self.sizes) > len(self.mnist):
                raise ValueError(
                    "Dataset sizes exceed the total number of available samples."
                )

    @staticmethod
    def _load_mnist(root, train, download, transform):
        return datasets.MNIST(root, train=train, download=download, transform=transform)

    @staticmethod
    def _combine_datasets(environments_datasets):
        combined_dataset = ConcatDataset(environments_datasets)
        indices = torch.randperm(len(combined_dataset))
        return Subset(combined_dataset, indices)

    def create_environments(self, combine_datasets=False):
        """
        Creates datasets for different environments.

        This method partitions the MNIST dataset according to the `sizes` provided and
        then applies two transformations: flipping labels with a probability defined
        by `p_flip_label` and flipping image colors with a probability defined by
        `e_s` (one for each environment). Additionally, images may be downsampled if
        the `downsample` flag is set to `True`.

        Parameters:
            combine_datasets (bool): Whether to combine the datasets from all environments
                                   into a single dataset. If `False`, a list of separate
                                   environment datasets is returned.

        Returns:
            list[CMNISTDataset] or Subset: A list of datasets for each environment.
                                    (if `combine_datasets` is `False`),
                                    or a single combined dataset (if `combine_datasets` is `True`).
        """

        environment_datasets = []
        last_index = 0

        if self.shuffle:
            indices = torch.randperm(len(self.mnist.targets))
            self.mnist.data = self.mnist.data[indices]
            self.mnist.targets = self.mnist.targets[indices]

        for environment, e in enumerate(self.e_s):
            images = self.mnist.data[last_index : last_index + self.sizes[environment]]
            labels = self.mnist.targets[
                last_index : last_index + self.sizes[environment]
            ]
            last_index += self.sizes[environment]

            if self.downsample:
                images = images[:, ::2, ::2]

            labels = (labels < 5).float()
            flip_label_mask = torch.bernoulli(
                torch.ones_like(labels) * self.p_flip_label
            )
            labels = torch.logical_xor(labels, flip_label_mask).float()

            flip_colors_mask = torch.bernoulli(torch.ones_like(labels) * e)
            colors = torch.logical_xor(labels, flip_colors_mask).float()

            images = torch.stack([images, images], dim=1)
            images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0
            images = images / 255

            dataset = CMNISTDataset(images, labels, environment)
            environment_datasets.append(dataset)

        if combine_datasets:
            return self._combine_datasets(environment_datasets)
        return environment_datasets


def show_image(image, label=None):
    image = image.permute((1, 2, 0)).cpu()
    image = torch.dstack([image, torch.zeros(image.shape[:-1])])
    plt.imshow(image)
    plt.axis("off")

    if label is not None:
        plt.title(f"Label: {label.item()}")


if __name__ == "__main__":

    cmnist = CMNIST(r"...")  # NOTE
    envs_dataset = cmnist.create_environments()

    for env_num, env in enumerate(envs_dataset):
        mask = (env[:][1] == 0).squeeze()

        high_numbers = env[:][0][mask]
        low_numbers = env[:][0][~mask]

        high_samples = make_grid(list(high_numbers[:25]), 5, 5)
        low_samples = make_grid(list(low_numbers[:25]), 5, 5)

        plt.subplot(2, 3, 1 + env_num)
        show_image(low_samples.cpu())
        plt.title(f"Env {env_num+1} - Low")

        plt.subplot(2, 3, 4 + env_num)
        show_image(high_samples.cpu())
        plt.title(f"Env {env_num+1} - High")
        plt.tight_layout()

    plt.suptitle("Training and Test Environments")
    plt.tight_layout()
    plt.show()
