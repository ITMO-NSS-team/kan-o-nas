import math
import os

import numpy as np
from torch.utils.data import Subset

from nas.data.time_series_preprocessing import multi_output_tensor


# from torchcnnbuilder.preprocess.time_series import multi_output_tensor


def create_test_train_sea_dataset(root, train_share=0.8, first_quartal_test_year=None):
    all_files = os.listdir(root)
    files = sorted(all_files)[::7]  # every 7th file: once a week;
    quartal_size = 13  # Q{1-4} is ~13 weeks
    if first_quartal_test_year:
        # train_size = int(round(all_files.index(f"osi_{first_quartal_test_year - 2}0101.npy") / 7))
        train_size = files.index(f"osi_{first_quartal_test_year - 2}0101.npy")
        print(files[train_size:])
    else:
        train_size = int(len(files) * train_share)

    pictures = [
        np.load(os.path.join(root, f))
        for f in files
    ]
    pictures = np.array(pictures)
    train_pictures, test_pictures = pictures[:train_size, :, :], pictures[train_size:, :, :]
    train_dataset = multi_output_tensor(train_pictures, pre_history_len=104, forecast_len=52)
    test_dataset = multi_output_tensor(test_pictures, pre_history_len=104, forecast_len=52)

    # Move to GPU
    # for d in [train_dataset, test_dataset]:
    #     d.tensors = [t.cuda() for t in d.tensors]

    sliced_test_dataset = [Subset(test_dataset, range(i, min(i + quartal_size, len(test_dataset))))
                           for i in range(0, len(test_dataset), quartal_size)]

    print(len(sliced_test_dataset))
    for i, t in enumerate(sliced_test_dataset):
        print(i)
    return train_dataset, sliced_test_dataset if first_quartal_test_year else [test_dataset]


if __name__ == '__main__':
    train_dataset, test_dataset = create_test_train_sea_dataset(r"C:\dev\aim\datasets\sea\laptev")
    print(train_dataset.tensors[0].shape)
    print(train_dataset.tensors[0].element_size() * train_dataset.tensors[0].nelement())
