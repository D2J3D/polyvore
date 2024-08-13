import torch
import torch.nn as nn


class ImageEmbeddingMapper(nn.Module):
    """
       A PyTorch module that maps the output of the Inception model to an embedding space.

       Args:
           inception_output_size (torch.Tensor): The size of output vector output from the Inception model.
           num_outputs (int): The size of the embedding space.

       Attributes:
           input_size (int): The size of the input to the module.
           num_outputs (int): The size of the output of the module.
           fc (nn.Linear): The fully connected layer that maps the input to the output.
    """
    def __init__(self, inception_output_size, num_outputs):
        super(ImageEmbeddingMapper, self).__init__()
        self.input_size = inception_output_size
        self.num_outputs = num_outputs
        self.initializer = nn.init.xavier_uniform_

        self.fc = nn.Linear(self.input_size, self.num_outputs)
        self.initializer(self.fc.weight)

    def forward(self, inception_output):
        return self.fc(inception_output)
