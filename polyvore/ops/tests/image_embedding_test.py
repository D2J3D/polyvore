import torch
from image_embedding import InceptionV3Embedding


model = InceptionV3Embedding(trainable=True, is_training=True)
images = torch.randn(8, 3, 299, 299)  # Example batch of images
output = model(images)
print(output.shape)  # Should output [8, 2048]
