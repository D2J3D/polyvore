import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, emb_margin):
        super().__init__()
        self.emb_margin = emb_margin

    def forward(self, norm_seq_embeddings, norm_image_embeddings):
        """
        Computes the contrastive loss for the given sequence and image embeddings.

        Args:
            seq_embeddings (torch.Tensor): A tensor of shape (sequence_count, embedding_size).
            img_embeddings (torch.Tensor): A tensor of shape (image_count, embedding_size).

        Returns:
            contrastive_loss (torch.Tensor): The computed contrastive loss.
        """
        scores = torch.mm(norm_seq_embeddings, norm_image_embeddings.T)
        diagonal = torch.diag(scores).unsqueeze(1)
        
        cost_s = torch.clamp(self.emb_margin - diagonal + scores, min=0.0)
        cost_im = torch.clamp(self.emb_margin - diagonal.T + scores, min=0.0)
        
        cost_s = cost_s - torch.diag(diagonal).diag()
        cost_im = cost_im - torch.diag(diagonal).diag()
        
        emb_batch_loss = cost_s.sum() + cost_im.sum()
        emb_batch_loss /= (norm_seq_embeddings.size(0) ** 2)
        
        return emb_batch_loss
