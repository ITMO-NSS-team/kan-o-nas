"""
Dataset implementations for in-gpu-memory caching for the specific requirements of full-NAS.
"""
import os
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class GpuCachedImageFolder(ImageFolder):
    def __init__(self, root, transform, target_transform: Optional[Callable] = None, eager: bool = False,
                 cache_before_transform: bool = False):
        if cache_before_transform:  # Let the parent class give untransformed images, then apply transform each time
            self.post_transform = transform
            underlying_transform = None
        else:  # Let the parent class give transformed images
            self.post_transform = None
            underlying_transform = transform
        super(GpuCachedImageFolder, self).__init__(root, transform=underlying_transform,
                                                   target_transform=target_transform)
        self.cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if eager:
            print("Caching all images into GPU…")
            for i in range(len(self)):
                _img = self[i]
            print("DONE caching all images into GPU")

    def __getitem__(self, index):
        if index in self.cache:
            return self.apply_post_transform(self.cache[index])

        # Get the image and label using the parent class method
        image, label = super(GpuCachedImageFolder, self).__getitem__(index)
        if isinstance(image, torch.Tensor):
            image = image.to(self.device)

        # Cache the image and label
        self.cache[index] = (image, label)
        return self.apply_post_transform((image, label))

    def apply_post_transform(self, item):
        image, label = item
        if self.post_transform is not None:
            image = self.post_transform(image)
        return image, label


# ## Inherited classes for particular datasets ## #

class EuroSAT(GpuCachedImageFolder):
    """RGB EuroSAT """

    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            eager: bool = False,
            cache_before_transform: bool = False
    ) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "eurosat")
        self._data_folder = os.path.join(self._base_folder, "2750")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        super().__init__(self._data_folder, transform=transform, target_transform=target_transform, eager=eager,
                         cache_before_transform=cache_before_transform)
        self.root = os.path.expanduser(root)

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder)

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self._base_folder, exist_ok=True)
        download_and_extract_archive(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
            download_root=self._base_folder,
            md5="c8fa014336c82ac7804f0398fcb19387",
        )


class CachedMNIST(GpuCachedImageFolder):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            eager: bool = False,
            train: bool = True,
            cache_before_transform: bool = False
    ) -> None:
        self.root = os.path.expanduser(root)
        self.folder = os.path.join(self.root, "MNISTFolder", "train" if train else "test")

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        super().__init__(self.folder, transform=transform, target_transform=target_transform, eager=eager,
                         cache_before_transform=cache_before_transform)
        self.root = os.path.expanduser(root)

    def _check_exists(self) -> bool:
        return os.path.exists(self.folder)


class CachedFashionMNIST(GpuCachedImageFolder):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            eager: bool = False,
            cache_before_transform: bool = False,
            train: bool = True
    ) -> None:
        self.root = os.path.expanduser(root)
        self.folder = os.path.join(self.root, "FashionMNISTFolder", "train" if train else "test")

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        super().__init__(self.folder, transform=transform, target_transform=target_transform, eager=eager,
                         cache_before_transform=cache_before_transform)
        self.root = os.path.expanduser(root)

    def _check_exists(self) -> bool:
        return os.path.exists(self.folder)


class CachedCIFAR10(GpuCachedImageFolder):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            eager: bool = False,
            train: bool = True,
            cache_before_transform: bool = False
    ) -> None:
        self.root = os.path.expanduser(root)
        self.folder = os.path.join(self.root, "CIFAR10Folder", "train" if train else "test")

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        super().__init__(self.folder, transform=transform, target_transform=target_transform, eager=eager,
                         cache_before_transform=cache_before_transform)
        self.root = os.path.expanduser(root)

    def _check_exists(self) -> bool:
        return os.path.exists(self.folder)
