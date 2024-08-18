# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Image embedding operations using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class InceptionV3Embedding(nn.Module):
    def __init__(self, 
                 trainable=True, 
                 dropout_keep_prob=0.8, 
                 use_batch_norm=True):
        super(InceptionV3Embedding, self).__init__()

        # Load pre-trained Inception V3 model
        self.inception_v3 = models.inception_v3(pretrained=True)

        # Make layers trainable if specified
        if not trainable:
            for param in self.inception_v3.parameters():
                param.requires_grad = False

        # Remove the default fully connected layer
        self.inception_v3.fc = nn.Identity()

        # Dropout layer
        self.dropout = nn.Dropout(p=1 - dropout_keep_prob)

        # Use batch normalization if specified
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(2048, eps=0.001, momentum=0.9997)

    def forward(self, images, is_training=True):
        # Set the model to train/eval mode
        self.inception_v3.train(is_training)

        # Forward pass through the InceptionV3 model
        features = self.inception_v3(images)

        # Apply dropout
        if is_training:
            features = self.dropout(features)

        # Apply batch normalization if specified
        if self.use_batch_norm:
            features = self.batch_norm(features)

        return features

# Example usage:
# model = InceptionV3Embedding(trainable=True, dropout_keep_prob=0.8, use_batch_norm=True)
# images = torch.randn(8, 3, 299, 299)  # Example input: batch of 8 images, 3 channels, 299x299 size
# embeddings = model(images, is_training=True)
