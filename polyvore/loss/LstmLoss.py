import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmLoss(nn.Module):
    def __init__(self, number_set_images, embedding_size, batch_size):
        super().__init__()
        self.number_set_images = number_set_images
        self.embedding_size = embedding_size
        self.batch_size = batch_size

    def forward(self, input_embeddings, target_embeedings, loss_mask):
        lstm_scores = torch.matmul(input_embeddings, target_embeedings.T)
        lstm_loss = F.cross_entropy(lstm_scores, torch.arange(
            lstm_scores.size(0)), reduction='sum')
        lstm_loss /= loss_mask.sum()
        return lstm_loss
