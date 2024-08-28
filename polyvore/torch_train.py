import torch
import torch.nn as nn
import torch.functional as F

from config.config import ModelConfig, TrainingConfig

from models.image_embedding_mapper import ImageEmbeddingMapper
from models.polyvore_model import PolyvoreModel

from loss.ContrastiveLoss import ContrastiveLoss
from loss.LstmLoss import LstmLoss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig()
    train_config = TrainingConfig()
    model = PolyvoreModel(config=model_config,
                          mode='train', train_inception=False)
    model.train()
    model.to(device)

    contrastive_criterion = ContrastiveLoss(model_config.emb_margin)
    lstm_criterion = LstmLoss(model_config.number_set_images, model_config.embedding_size, model_config.batch_size)

    for i, batch in enumerate(train_loader):
        pass
    # total_loss = 0.5 * contrastive_loss + 0.5 * lstm_loss

