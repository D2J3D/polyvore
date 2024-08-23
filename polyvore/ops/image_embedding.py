import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class InceptionV3Embedding(nn.Module):
    def __init__(self, 
                 trainable=True, 
                 is_training=True, 
                 weight_decay=0.00004, 
                 dropout_keep_prob=0.8, 
                 use_batch_norm=True, 
                 add_summaries=True):
        super(InceptionV3Embedding, self).__init__()

        self.trainable = trainable
        self.is_training = is_training
        self.weight_decay = weight_decay
        self.dropout_keep_prob = dropout_keep_prob
        self.use_batch_norm = use_batch_norm
        self.add_summaries = add_summaries

        # Load the pre-trained Inception V3 model
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

        # Freeze the parameters if the model is not trainable
        if not self.trainable:
            for param in self.inception.parameters():
                param.requires_grad = False

        # Replace the last fully connected layer with an identity layer
        self.inception.fc = nn.Identity()

        # Set the dropout layer
        self.dropout = nn.Dropout(p=1 - self.dropout_keep_prob)

        # Set batch normalization if required
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(2048, eps=0.001, momentum=0.9997)
        else:
            self.batch_norm = None

    def forward(self, images):
        # Forward pass through the Inception model
        x = self.inception(images)

        # Extract only the logits from the InceptionOutputs
        x = x.logits

        # Apply dropout
        if self.is_training:
            x = self.dropout(x)

        # Apply batch normalization
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        # Optionally add summaries for visualization (e.g., using TensorBoard)
        if self.add_summaries:
            self._add_summaries(x)

        return x

    def _add_summaries(self, x):
        # Example summary addition using TensorBoard or other logging mechanisms
        # Replace with actual summary code if needed.
        print(f"Summary of activations: {x}")


