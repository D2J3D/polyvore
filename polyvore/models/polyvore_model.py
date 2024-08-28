import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops

from models.image_embedding_mapper import ImageEmbeddingMapper


class PolyvoreModel(nn.Module):
    def __init__(self, config, mode, train_inception=False):
        """
        Basic setup.
            Args:
              config: Object containing configuration parameters.
              mode: "train", "eval" or "inference".
              train_inception: Whether the inception submodel variables are trainable.
        """
        super().__init__()
        # Set essential parameters
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Set device
        self.device_id = 0 if torch.cuda.is_available() else -1
        torch.cuda.set_device(self.device_id)
        self.device = torch.device(
            f'cuda:{self.device_id}' if self.device_id >= 0 else 'cpu')

        # Embedding mapper model
        self.mapper = None

        # Model
        # Forward LSTM.
        self.f_lstm_cell = nn.LSTMCell(
            input_size=self.config.embedding_size, hidden_size=self.config.num_lstm_units)
        self.f_dropout = nn.Dropout(p=1-self.config.lstm_dropout_keep_prob)

        # Backward LSTM.
        self.b_lstm_cell = nn.LSTMCell(
            input_size=self.config.embedding_size, hidden_size=self.config.num_lstm_units)
        self.b_dropout = nn.Dropout(p=1-self.config.lstm_dropout_keep_prob)
        
        # Linear layers   
        self.f_fc = nn.Linear(self.config.num_lstm_units,
                              self.config.embedding_size)
        self.b_fc = nn.Linear(self.config.num_lstm_units,
                              self.config.embedding_size)

    def _is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def forward(self, image_seqs, input_mask, loss_mask, cap_seqs, cap_mask):
        """
            Forward method of the model.
        """

        # Building embeddings
        # set_ids, image_seqs, image_ids, input_mask, loss_mask, cap_seqs, cap_mask, likes = self._build_inputs()
        image_embeddings, rnn_image_embeddings = self._build_image_embeddings(
            image_seqs)
        embedding_map, seq_embeddings = self._build_seq_embeddings(
            cap_seqs, cap_mask)

        # Normalize img and seq embeddings
        norm_image_embeddings = F.normalize(
            image_embeddings, dim=2, eps=1e-12)
        norm_seq_embeddings = F.normalize(
            seq_embeddings, dim=2, eps=1e-12)

        # Apply padding to seq embeddings
        num_paddings = max(0, self.config.number_set_images -
                           norm_seq_embeddings.size(1))
        if num_paddings > 0:
            padding = torch.zeros((norm_seq_embeddings.size(0), num_paddings))
            norm_seq_embeddings = torch.cat(
                [norm_seq_embeddings, padding], dim=-1)

        # reshape the embeddings for calculating constractive loss function.
        if self.mode != "inference":
            norm_image_embeddings = norm_image_embeddings.reshape(
                self.config.number_set_images * self.config.batch_size, self.config.embedding_size)
            norm_seq_embeddings = norm_seq_embeddings.reshape(
                self.config.number_set_images * self.config.batch_size, self.config.embedding_size)

        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = torch.sum(input_mask, dim=1)
        lstm_outputs, _ = nn.utils.rnn.pack_padded_sequence(
            rnn_image_embeddings, sequence_length, batch_first=True)
        lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_outputs, batch_first=True)
        f_lstm_outputs, b_lstm_outputs = self.f_lstm_cell(
            lstm_outputs), self.b_lstm_cell(lstm_outputs)
        f_lstm_outputs = self.f_dropout(f_lstm_outputs)
        b_lstm_outputs = self.b_dropout(b_lstm_outputs)

        # Stack batches vertically.
        f_lstm_outputs = f_lstm_outputs.view(-1, self.config.num_lstm_units)
        if self.mode == "inference":
            b_lstm_outputs = lstm_outputs[1]
        else:
            b_lstm_outputs = torch.flip(lstm_outputs[1], [1])

        b_lstm_outputs = torch.view(-1, self.config.num_lstm_units)

        f_input_embeddings = self.f_fc(f_lstm_outputs)
        b_input_embeddings = self.b_fc(b_lstm_outputs)

        return f_input_embeddings, b_input_embeddings, norm_image_embeddings, norm_seq_embeddings
