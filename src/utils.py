from collections import defaultdict

import torch
import matplotlib.pyplot as plt


def get_n_samples_per_class(dataset, n_classes=10, n=5):
    """
    Retrieves `n` samples per class from the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The PyTorch dataset to sample from.
        The dataset is expected to return (image, label) tuples when iterated over.
        n_classes (int, optional): The number of unique classes/labels in the dataset.
                                   Default is 10.
        n (int, optional): The number of samples to retrieve for each class. Default is 5.

    Returns:
        list: A list of lists, where each inner list contains `n` images for a
              corresponding label. The outer list has length `n_classes` (one list per class).
    """
    samples_per_class = defaultdict(list)

    for image, label in dataset:
        if len(samples_per_class[label.item()]) < n:
            samples_per_class[label.item()].append(image)

        if all([len(samples_per_class[i]) == n for i in range(n_classes)]):
            break

    samples_per_class = [samples_per_class[i] for i in range(n_classes)]
    return samples_per_class


def show_image(image, label=None):
    image = image.permute((1, 2, 0)).cpu()
    image = torch.dstack([image, torch.zeros(image.shape[:-1])])
    plt.imshow(image)
    plt.axis("off")

    if label is not None:
        plt.title(f"Label: {label.item()}")


COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (210, 177, 0),
    (220, 69, 0),
    (80, 135, 0),
    (15, 166, 0),
    (50, 90, 0),
    (192, 237, 0),
    (45, 36, 0),
    (129, 8, 0),
)
