# Copyright 2017 Xintong Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input operations converted from TensorFlow to PyTorch."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceExampleDataset(Dataset):
    def __init__(self, data, set_id, image_feature, image_index, caption_feature, number_set_images):
        self.data = data
        self.set_id = set_id
        self.image_feature = image_feature
        self.image_index = image_index
        self.caption_feature = caption_feature
        self.number_set_images = number_set_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        set_id = item[self.set_id]
        likes = item.get('likes', 0)

        encoded_images = []
        for i in range(self.number_set_images):
            encoded_images.append(item.get(f'{self.image_feature}/{i}', ''))

        captions = torch.tensor(item[self.caption_feature])
        image_ids = torch.tensor(item[self.image_index])

        return set_id, encoded_images, image_ids, captions, likes


def prefetch_input_data(file_pattern, is_training, batch_size, num_workers=1):
    """Prefetches data for loading into PyTorch DataLoader."""
    from glob import glob

    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(glob(pattern))
    if not data_files:
        raise FileNotFoundError(f"Found no input files matching {file_pattern}")
    
    print(f"Prefetching values from {len(data_files)} files matching {file_pattern}")

    dataset = SequenceExampleDataset(data=data_files, set_id="set_id", image_feature="image_feature", 
                                     image_index="image_index", caption_feature="caption_feature", 
                                     number_set_images=5)  # Adjust `number_set_images` based on your needs

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=num_workers)

    return data_loader


def batch_with_dynamic_pad(batch_data):
    """Batch and pad sequences dynamically for PyTorch."""
    # Unpack the batch data
    set_ids, images, image_ids, captions, likes = zip(*batch_data)

    # Calculate max lengths for padding
    max_image_seq_length = max(len(ids) for ids in image_ids)
    max_caption_length = max(len(caption) for caption in captions)

    # Padding
    padded_images = []
    for img_set in images:
        padded_set = [img for img in img_set]
        while len(padded_set) < max_image_seq_length:
            padded_set.append('')
        padded_images.append(padded_set)

    padded_captions = []
    mask = []
    for caption in captions:
        pad_len = max_caption_length - len(caption)
        padded_captions.append(torch.cat([caption, torch.zeros(pad_len, dtype=torch.long)]))
        mask.append(torch.cat([torch.ones(len(caption)), torch.zeros(pad_len)]))

    set_ids = torch.tensor(set_ids)
    padded_images = torch.tensor(padded_images)
    padded_captions = torch.stack(padded_captions)
    mask = torch.stack(mask)
    likes = torch.tensor(likes)

    return set_ids, padded_images, padded_captions, mask, likes
