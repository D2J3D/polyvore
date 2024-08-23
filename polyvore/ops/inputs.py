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

"""Input ops adapted for PyTorch."""

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

def parse_sequence_example(serialized, set_id, image_feature,
                           image_index, caption_feature, number_set_images):
    """Parses a serialized example into a set of images and captions.

    Args:
        serialized: Serialized example (as dict or other format).
        set_id: Key for the set ID in the serialized data.
        image_feature: Key prefix for image data in the serialized data.
        image_index: Key for the image indices in the serialized data.
        caption_feature: Key for caption data in the serialized data.
        number_set_images: Number of images in a set.

    Returns:
        set_id: Set id of the outfit.
        encoded_images: A list containing all encoded images in the outfit.
        image_ids: Image ids of the items in the outfit.
        captions: A tensor with dynamically specified length.
        likes: Number of likes of the outfit.
    """
    context_features = {}
    context_features[set_id] = serialized[set_id]
    context_features['likes'] = serialized.get('likes', 0)
    
    encoded_images = []
    for i in range(number_set_images):
        image_key = f"{image_feature}/{i}"
        encoded_images.append(serialized.get(image_key, ''))
    
    image_ids = serialized[image_index]
    captions = torch.tensor(serialized[caption_feature], dtype=torch.int64)
    
    return context_features[set_id], encoded_images, image_ids, captions, context_features['likes']


class CustomDataset(data.Dataset):
    def __init__(self, serialized_data, set_id, image_feature, image_index, caption_feature, number_set_images):
        self.data = serialized_data
        self.set_id = set_id
        self.image_feature = image_feature
        self.image_index = image_index
        self.caption_feature = caption_feature
        self.number_set_images = number_set_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        serialized = self.data[idx]
        return parse_sequence_example(serialized, self.set_id, self.image_feature,
                                      self.image_index, self.caption_feature, self.number_set_images)


def batch_with_dynamic_pad(images_and_captions, batch_size, add_summaries=True):
    """Batches input images and captions with dynamic padding.

    Args:
        images_and_captions: A dataset with image and caption data.
        batch_size: Batch size.
        add_summaries: If true, add summaries.

    Returns:
        Batches of images, captions, masks, etc.
    """
    dataloader = DataLoader(images_and_captions, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    set_ids, encoded_images, image_ids, captions, likes = zip(*batch)

    captions_padded = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    masks = (captions_padded != 0).long()
    
    return set_ids, encoded_images, image_ids, captions_padded, masks, likes
