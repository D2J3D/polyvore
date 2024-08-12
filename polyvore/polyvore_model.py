import torch
import torch.nn as nn

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops


class PolyvoreModel(nn.Module):
    def __init__(self, config, mode, train_inception=False):
        """Basic setup.
            Args:
              config: Object containing configuration parameters.
              mode: "train", "eval" or "inference".
              train_inception: Whether the inception submodel variables are trainable.
            """
        super().__init__()
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        self.device_id = 0 if torch.cuda.is_available() else -1
        torch.cuda.set_device(self.device_id)
        self.device = torch.device(f'cuda:{self.device_id}' if self.device_id >= 0 else 'cpu')

        # A float32 Tensor with shape
        # [batch_size, num_images, height, width, channels].
        # num_images is the number of images in one outfit, default is 8.
        self.images = None

        # Forward RNN input and target sequences.
        # An int32 Tensor with shape [batch_size, padded_length].
        self.f_input_seqs = None
        # An int32 Tensor with shape [batch_size, padded_length].
        self.f_target_seqs = None

        # Backward RNN input and target sequences.
        # An int32 Tensor with shape [batch_size, padded_length].
        self.b_input_seqs = None
        # An int32 Tensor with shape [batch_size, padded_length].
        self.b_target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # Image caption sequence and masks.
        # An int32 Tensor with shape [batch_size, num_images, padded_length].
        self.cap_seqs = None
        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.cap_mask = None

        # Caption sequence embeddings, we use simple bag of word model.
        # A float32 Tensor with shape [batch_size, num_images, embedding_size].
        self.seq_embeddings = None

        # Image embeddings in the joint visual-semantic space
        # A float32 Tensor with shape [batch_size, num_images, embedding_size].
        self.image_embeddings = None

        # Image embeddings in the RNN output/prediction space.
        self.rnn_image_embeddings = None

        # Word embedding map.
        self.embedding_map = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # Forward and backward RNN loss.
        # A float32 Tensor with shape [batch_size * padded_length].
        self.forward_losses = None
        # A float32 Tensor with shape [batch_size * padded_length].
        self.backward_losses = None
        # RNN loss, forward + backward.
        self.lstm_losses = None

        # Loss mask for lstm loss.
        self.loss_mask = None

        # Visual Semantic Embedding loss.
        # A float32 Tensor with shape [batch_size * padded_length].
        self.emb_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_weights = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

        # Some output for debugging purposes .
        self.target_embeddings = None
        self.input_embeddings = None
        self.set_ids = None
        self.f_lstm_state = None
        self.b_lstm_state = None
        self.lstm_output = None
        self.lstm_xent_loss = None

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

        self.images = image_seqs
        self.input_mask = input_mask
        self.loss_mask = loss_mask
        self.cap_seqs = cap_seqs
        self.cap_mask = cap_mask
        self.set_ids = set_ids

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings
              in visual semantic joint space and RNN prediction space.

            Inputs:
              self.images

            Outputs:
              self.image_embeddings
              self.rnn_image_embeddings
        """

        # Reshape 5D image tensor
        images = torch.reshape(self.images, [-1,
                                             self.config.image_height,
                                             self.config.image_width,
                                             3])
        inception_output = image_embedding.inception_v3(
            images,
            trainable=self.train_inception,
            is_training=self.is_training())
