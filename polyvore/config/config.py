from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Sets the default model hyperparameters."""
    # Image format ("jpeg" or "png").
    image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    values_per_input_shard = 135
    # Minimum number of shards to keep in the input queue.
    input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    num_input_reader_threads = 1  # potentially odd param

    # Name of the SequenceExample context feature containing set ids.
    set_id_name = "set_id"
    # Name of the SequenceExample feature list containing captions and images.
    image_feature_name = "images"
    image_index_name = "image_index"
    caption_feature_name = "caption_ids"

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    vocab_size = 2757

    # Number of threads for image preprocessing.
    num_preprocess_threads = 1  # potentially odd param

    # File containing an Inception v3 checkpoint to initialize the variables
    # of the Inception model. Must be provided when starting training for the
    # first time.
    inception_checkpoint_file = None

    # Dimensions of Inception v3 input images.
    image_height = 299
    image_width = 299

    # Scale used to initialize model variables.
    initializer_scale = 0.08

    # LSTM input and output dimensionality, respectively. embedding_size is also
    # the embedding size in the visual-semantic joint space.
    embedding_size = 512
    num_lstm_units = 512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    lstm_dropout_keep_prob = 0.7

    # Largest number of images in a fashion set.
    number_set_images = 8

    # Margin for the embedding loss.
    emb_margin = 0.2

    batch_size = 10

    # Balance factor of all losses.
    emb_loss_factor = 1.0  # VSE loss
    f_rnn_loss_factor = 1.0  # Forward LSTM
    b_rnn_loss_factor = 1.0  # Backward LSTM, might give it a lower weight
    # because it is harder to predict backward than forward in our senario.


@dataclass
class TrainingConfig:
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    num_examples_per_epoch = 17316

    # Optimizer for training the model.
    optimizer = "SGD"

    # Learning rate for the initial phase of training.
    # by the FLAGS in train.py
    initial_learning_rate = 0.2

    learning_rate_decay_factor = 0.5
    num_epochs_per_decay = 2.0

    # If not None, clip gradients to this value.
    clip_gradients = 5.0

    # How many model checkpoints to keep.
    max_checkpoints_to_keep = 10
