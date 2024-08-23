# Copyright 2024 Your Name. All Rights Reserved.
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

"""Helper functions for image preprocessing using PyTorch."""

import torch
import torchvision.transforms as transforms
from PIL import Image
import io

def distort_image(image):
    """Perform random distortions on an image.

    Args:
        image: A PIL Image with shape [height, width, 3].

    Returns:
        distorted_image: A PIL Image with shape [height, width, 3].
    """
    # Randomly flip horizontally.
    transform = transforms.RandomHorizontalFlip(p=0.5)
    image = transform(image)

    return image


def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=299,
                  resize_width=299,
                  image_format="jpeg",
                  image_idx=0):
    """Decode an image, resize and apply random distortions.

    Args:
        encoded_image: Byte string containing the image.
        is_training: Boolean; whether preprocessing for training or eval.
        height: Height of the output image.
        width: Width of the output image.
        resize_height: If > 0, resize height before crop to final dimensions.
        resize_width: If > 0, resize width before crop to final dimensions.
        image_format: "jpeg" or "png".
        image_idx: Image index of the image in an outfit.

    Returns:
        A Tensor of shape [3, height, width] with values in [-1, 1].

    Raises:
        ValueError: If image_format is invalid.
    """
    # Decode the image from a byte string.
    if image_format.lower() not in ["jpeg", "png"]:
        raise ValueError(f"Invalid image format: {image_format}")

    image = Image.open(io.BytesIO(encoded_image))

    # Convert the image to RGB (if needed) and to a float32 tensor.
    transform_list = []

    if resize_height > 0 and resize_width > 0:
        transform_list.append(transforms.Resize((resize_height, resize_width)))

    if is_training:
        transform_list.append(transforms.RandomCrop((height, width)))
        transform_list.append(transforms.Lambda(distort_image))
    else:
        transform_list.append(transforms.CenterCrop((height, width)))

    # Convert to tensor and normalize to [-1, 1]
    transform_list.extend([
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
    ])

    transform = transforms.Compose(transform_list)
    image = transform(image)

    return image


# Пример использования
with open('example.jpg', 'rb') as f:
    encoded_image = f.read()

# Тестирование
processed_image = process_image(encoded_image, is_training=True, height=224, width=224)
print(processed_image.shape)  # Должно вывести torch.Size([3, 224, 224])
