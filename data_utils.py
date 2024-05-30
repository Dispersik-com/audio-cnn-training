import os
import random

import torch
from audio_utils import extract_mfcc_segment, normalize_mfcc
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataset(audio_data_dict, output_dir_name=None,
                   resize=False, n_mfcc=20, normalized_mfcc=False):
    stacks = []
    classes = []
    for file_name, segments_list in audio_data_dict.items():
        tensors = []
        classes.append(file_name)
        for segment in segments_list[0]:
            mfcc = extract_mfcc_segment(segment, n_mfcc=n_mfcc)
            if normalized_mfcc:
                mfcc = normalize_mfcc(mfcc)
            tensors.append(torch.from_numpy(mfcc))
        temp_stack = torch.stack(tensors)
        stacks.append(temp_stack)

    min_length_data = 0
    mfcc_size_dict = {}
    output_dir_name = "dataset" if output_dir_name is None else output_dir_name

    if not os.path.exists(f"{output_dir_name}"):
        os.mkdir(f"{output_dir_name}")

    if not os.path.exists(f"{output_dir_name}/pths"):
        os.mkdir(f"{output_dir_name}/pths")

    if resize:
        min_length_data = min(stack.size(0) for stack in stacks)
        stacks = [stack[:min_length_data] for stack in stacks]

    for i, stack_tensors in enumerate(stacks):
        mfcc_size_dict[classes[i]] = stacks[i][0].size()
        torch.save(stack_tensors, f"{output_dir_name}/pths/{classes[i]}.pth")

    # mfcc_size = stacks[0][0].size()
    return stacks, mfcc_size_dict, min_length_data


def load_data(file_paths: list):
    datasets = []
    for file_path, label in file_paths:
        data = torch.load(file_path)
        labels = torch.full((len(data),), label, dtype=torch.long)
        dataset = MyDataset(data, labels)
        datasets.append(dataset)

    full_dataset = ConcatDataset(datasets)
    return full_dataset


def split_data(dataset, size=0.8, batch_size=32, num_workers=3):
    train_size = int(size * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
