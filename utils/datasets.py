import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and labels from NumPy files.

    This dataset handles loading .npy files for both images and labels,
    with support for optional transforms and automatic NaN/Inf replacement.

    Args:
        image_paths: List of paths to image .npy files.
        label_data: Either a list of paths to label .npy files, or a list of
            preloaded numpy arrays.
        transform: Optional callable transform to apply to (image, label) pairs.

    Returns:
        When indexed, returns a tuple of (image, label, image_path) where:
        - image: torch.Tensor of shape matching the loaded .npy file
        - label: torch.Tensor of shape matching the loaded label
        - image_path: str path to the original image file

    Example:
        >>> image_paths = ["data/img1.npy", "data/img2.npy"]
        >>> label_paths = ["data/lbl1.npy", "data/lbl2.npy"]
        >>> dataset = NumpyDataset(image_paths, label_paths)
        >>> image, label, path = dataset[0]
    """

    def __init__(
        self,
        image_paths: List[str],
        label_data: Union[List[str], List[np.ndarray]],
        transform: Optional[callable] = None
    ):
        self.image_paths = image_paths
        self.label_data = label_data  # Accept either paths or preloaded arrays
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image = np.load(self.image_paths[idx])

        # Check if label_data contains paths or arrays
        if isinstance(self.label_data[idx], str):
            label = np.load(self.label_data[idx])
        else:
            label = self.label_data[idx]

        image = self.replace_nans_in_array(image)
        label = self.replace_nans_in_array(label)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations if specified
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, self.image_paths[idx]

    @staticmethod
    def replace_nans_in_array(arr: np.ndarray) -> np.ndarray:
        """
        Replace NaN and Inf values in an array with 0.

        Args:
            arr: Input numpy array that may contain NaN or Inf values.

        Returns:
            Array with NaN and Inf values replaced by 0.
        """
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0
        return arr


def load_dataset_json(json_path: str) -> Dict[str, Any]:
    """
    Load dataset paths from a JSON file.

    Reads a JSON file containing dataset configuration, typically including
    paths to training, validation, and testing data splits.

    Args:
        json_path: Path to the JSON file containing dataset information.

    Returns:
        Dictionary containing the parsed JSON data with dataset paths.

    Example:
        >>> dataset_json = load_dataset_json("Dataset/dataset.json")
        >>> train_paths = dataset_json['training']
        >>> val_paths = dataset_json['validation']
        >>> test_paths = dataset_json['testing']
    """
    with open(json_path, 'r') as file:
        dataset_json = json.load(file)
    return dataset_json


def prepare_test_loader(
    test_image_paths: List[str],
    batch_size: int,
    dummy_label_dir: str = "./model/temp_labels/",
    num_workers: int = 4
) -> DataLoader:
    """
    Prepare a DataLoader for test/inference with dummy labels.

    Creates dummy (zero-filled) labels for test images since labels are not
    available during inference. This allows using the same NumpyDataset class
    for both training and inference.

    Args:
        test_image_paths: List of paths to test image .npy files.
        batch_size: Number of samples per batch.
        dummy_label_dir: Directory to store dummy label files (default: "./model/temp_labels/").
        num_workers: Number of worker processes for data loading (default: 4).

    Returns:
        A PyTorch DataLoader configured for test/inference.

    Example:
        >>> test_paths = ["data/test1.npy", "data/test2.npy"]
        >>> test_loader = prepare_test_loader(test_paths, batch_size=12)
        >>> for images, labels, paths in test_loader:
        ...     predictions = model(images)
    """
    if not os.path.exists(dummy_label_dir):
        os.makedirs(dummy_label_dir)

    test_labels = []
    for image_path in test_image_paths:
        dummy_label_path = os.path.join(dummy_label_dir, os.path.basename(image_path).replace('.npy', '_label.npy'))
        np.save(dummy_label_path, np.zeros_like(np.load(image_path)))
        test_labels.append(dummy_label_path)

    test_dataset = NumpyDataset(test_image_paths, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader
