from PIL import Image
from torch.utils import data
from typing import Tuple, Any


class Classification(data.Dataset):

    def __init__(self, dataset, transform=None, target_transform=None) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.dataset[index]
        sample, label = sample['image'], sample['label']
        if isinstance(sample, str):
            sample = Image.open(sample)

        sample = sample.convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            # print("Label:", label)
            label = self.target_transform(label)

        return sample, label

    def __len__(self) -> int:
        return self.length
