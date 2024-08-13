import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops
from ops.image_embedding_mapping import ImageEmbeddingMapper


class PolyvoreModel(nn.Module):
    def __init__(self, config, mode, train_inception=False):
        """Basic setup.
            Args:
              config: Object containing configuration parameters.
              mode: "train", "eval" or "inference".
              train_inception: Whether the inception submodel variables are trainable.
            """
        super(PolyvoreModel, self).__init__()
        # Set essential parameters
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Set device
        self.device_id = 0 if torch.cuda.is_available() else -1
        torch.cuda.set_device(self.device_id)
        self.device = torch.device(f'cuda:{self.device_id}' if self.device_id >= 0 else 'cpu')

    def _is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def _process_image(self, encoded_image, image_idx=0):
        """Decodes and processes an image string.

        Args:
          encoded_image: A scalar string Tensor; the encoded image.
          thread_id: Preprocessing thread id used to select the ordering of color
            distortions. Not used in our model.
          image_idx: Index of the image in an outfit. Only used for summaries.
        Returns:
          A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              image_format=self.config.image_format,
                                              image_idx=image_idx)

    def _build_inputs(self):
        """Input prefetching, preprocessing and batching.

            Outputs:
              Inputs of the model.
        """
        if self.mode == "inference":
            with torch.no_grad():
                # In inference mode, images and inputs are fed via tensors
                image_feed = torch.tensor([], dtype=torch.string)
                # Process image and insert batch dimensions.
                image_feed = self._process_image(image_feed)

                input_feed = torch.tensor([], dtype=torch.int64)

                # Process image and insert batch dimensions.
                image_seqs = image_feed.unsqueeze(0)
                cap_seqs = input_feed.unsqueeze(1)

                # No target sequences or input mask in inference mode.
                input_mask = torch.tensor([1, 8], dtype=torch.int64)

                cap_mask = None
                loss_mask = None
                set_ids = None
        else:
            # TODO change after writing dataloaders
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion. But we only
            # use one thread in our Polyvore model. likes are not used.
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                set_id, encoded_images, image_ids, captions, likes = (
                    input_ops.parse_sequence_example(
                        serialized_sequence_example,
                        set_id=self.config.set_id_name,
                        image_feature=self.config.image_feature_name,
                        image_index=self.config.image_index_name,
                        caption_feature=self.config.caption_feature_name,
                        number_set_images=self.config.number_set_images))

                images = []
                for i in range(self.config.number_set_images):
                    images.append(self.process_image(encoded_images[i], image_idx=i))

                images_and_captions.append([set_id, images, image_ids, captions, likes])

            # Batch inputs.
            queue_capacity = (5 * self.config.num_preprocess_threads *
                              self.config.batch_size)

            (set_ids, image_seqs, image_ids, input_mask,
             loss_mask, cap_seqs, cap_mask, likes) = (
                input_ops.batch_with_dynamic_pad(images_and_captions,
                                                 batch_size=self.config.batch_size,
                                                 queue_capacity=queue_capacity))

        # self.images = image_seqs
        # self.input_mask = input_mask
        # self.loss_mask = loss_mask
        # self.cap_seqs = cap_seqs
        # self.cap_mask = cap_mask
        # self.set_ids = set_ids

        return image_seqs, input_mask, loss_mask, cap_seqs, cap_mask, set_ids

    def _build_image_embeddings(self, images):
        """Builds the image model subgraph and generates image embeddings
              in visual semantic joint space and RNN prediction space.

            Inputs:
              self.images (image_seqs)

            Outputs:
              self.image_embeddings
              self.rnn_image_embeddings
        """

        # Reshape 5D image tensor
        images = torch.reshape(images, [-1,
                                        self.config.image_height,
                                        self.config.image_width,
                                        3])
        inception_output = image_embedding.inception_v3(
            images,
            trainable=self.train_inception,
            is_training=self.is_training())
        mapper = ImageEmbeddingMapper(inception_output.shape[1], self.config.embedding_size)

        with torch.no_grad():
            image_embeddings = mapper(inception_output)
            rnn_image_embeddings = mapper(inception_output)

        image_embeddings = torch.reshape(image_embeddings,
                                         [images.shape[0],
                                          -1,
                                          self.config.embedding_size])
        rnn_image_embeddings = torch.reshape(rnn_image_embeddings,
                                             [images[0],
                                              -1,
                                              self.config.embedding_size])

        return image_embeddings, rnn_image_embeddings

    def _build_seq_embeddings(self, cap_seqs, cap_mask):
        """Builds the input sequence embeddings.

            Inputs:
              self.input_seqs

            Outputs:
              self.seq_embeddings
              self.embedding_map
         """
        embedding_map = torch.empty(self.config.vocab_size, self.config.embedding_size).uniform_(
            -self.config.initializer_scale, self.config.initializer_scale)
        seq_embeddings = embedding_map[cap_seqs]
        if self.mode != "inference":
            seq_embeddings = torch.bmm(cap_mask.to('float').unsqueeze(2), seq_embeddings)
            seq_embeddings = seq_embeddings.squeeze(dim=2)

        # self.embedding_map = embedding_map
        # self.seq_embeddings = seq_embeddings

        return embedding_map, seq_embeddings

    def forward(self, image_embeddings, seq_embeddings):
        """
        Draft version working on prefetched embeddings.
        """
        norm_image_embeddings = F.normalize(image_embeddings, 2)
        norm_seq_embeddings = F.normalize(seq_embeddings, 2)
        padding_length = self.config.number_set_images - norm_seq_embeddings.size(1)
        norm_seq_embeddings = torch.cat(
            (norm_seq_embeddings, norm_seq_embeddings.new_zeros(norm_seq_embeddings.size(0), padding_length)), dim=1)

    def _build_model(self):
        """
        Builds the model
        """
        norm_image_embeddings = F.normalize(self.image_embeddings, 2)
        norm_seq_embeddings = F.normalize(self.seq_embeddings, 2)
        padding_length = self.config.number_set_images - norm_seq_embeddings.size(1)
        norm_seq_embeddings = torch.cat(
            (norm_seq_embeddings, norm_seq_embeddings.new_zeros(norm_seq_embeddings.size(0), padding_length)), dim=1)
        if self.mode == "inference":
            pass
        else:
            # Compute losses for joint embedding.
            # Only look at the captions that have length >= 2.
            emb_loss_mask = torch.where(torch.sum(self.cap_mask, dim=2) > 1, True, False)
            # Image mask is padded it to max length.
            max_sequence_length = self.config.number_set_images
            padding_length = max_sequence_length - emb_loss_mask.size(-1)
            emb_loss_mask = torch.cat((emb_loss_mask, torch.zeros(emb_loss_mask.size(0), padding_length)), dim=-1)
