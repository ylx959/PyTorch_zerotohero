"""
 modular script
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch


numworkers=os.cpu_count()


def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers=numworkers):
  """Create PyTorch DataLoaders for training and testing datasets.

    This function loads data from the specified training and testing directories,
    applies the given transforms, and constructs DataLoader instances for each set.

    Args:
        train_dir (str or Path): Path to the training data directory.
        test_dir (str or Path): Path to the testing data directory.
        transform (callable): Transformations to apply to the datasets.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
            list[str]
        ]:
            A tuple containing the training DataLoader, testing DataLoader,
            and a list of class names.
  """
  # Create datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Create dataloaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=numworkers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=numworkers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names