import logging

from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, BatchNormalization, Dropout,
    Bidirectional, LSTM, Dense, TimeDistributed, Add, MultiHeadAttention
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.legacy import Adam
import pickle
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa
from tf2crf import CRF, ModelWithCRFLoss

import os, tensorflow as tf
from scripts.available_gpus import gpus_to_visible_devices_string
# TODO: Will need to think about how to handle this when users are able to pick
#       which specific GPUs to use. This may need to be set elsewhere, or still
#       set here, but using a different function
os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_visible_devices_string()
tf.config.experimental.enable_tensor_float_32_execution(False)

logger = logging.getLogger(__name__)

# Mapping nucleotides to integers
nucleotide_to_id = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}

def encode_sequence(sequence):
    """Convert nucleotide sequence to list of integers."""
    return [nucleotide_to_id[base] for base in sequence]

class DynamicPaddingDataGenerator(Sequence):
    def __init__(self, X, Y, batch_size, label_binarizer):
        self.X = [encode_sequence(seq) for seq in X]
        self.Y = [label_binarizer.transform(labels) for labels in Y]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_Y = self.Y[idx * self.batch_size:(idx + 1) * self.batch_size]
        max_len = max(len(x) for x in batch_X)
        X_padded = pad_sequences(batch_X, maxlen=max_len, padding="post", value=0)
        Y_padded = pad_sequences(batch_Y, maxlen=max_len, padding="post", value=0)
        return X_padded, Y_padded

def ont_read_annotator(
    vocab_size,
    embedding_dim,
    num_labels,
    conv_layers=3,
    conv_filters=260,
    conv_kernel_size=25,
    lstm_layers=1,
    lstm_units=128,
    bidirectional=True,
    crf_layer=True,
    attention_heads=0,
    dropout_rate=0.35,
    regularization=0.01,
    learning_rate=0.01
):
    inputs = Input(shape=(None,), dtype="int32", name="input_tokens")
    x = Embedding(vocab_size, embedding_dim, name="embedding")(inputs)
    for i in range(conv_layers):
        x = Conv1D(
            filters=conv_filters,
            kernel_size=conv_kernel_size,
            activation="relu",
            padding="same",
            kernel_regularizer=l2(regularization),
            name=f"conv1d_{i}"
        )(x)
        x = BatchNormalization(name=f"batchnorm_{i}")(x)
        x = Dropout(dropout_rate, name=f"dropout_conv_{i}")(x)
    for i in range(lstm_layers):
        lstm_layer = LSTM(
            lstm_units if i == 0 else lstm_units // 2,
            return_sequences=True,
            kernel_regularizer=l2(regularization),
            recurrent_regularizer=l2(regularization),
            name=f"lstm_{i}"
        )
        x = Bidirectional(lstm_layer, name=f"bilstm_{i}")(x) if bidirectional else lstm_layer(x)
        x = Dropout(dropout_rate, name=f"dropout_lstm_{i}")(x)
    if attention_heads > 0:
        attention_out = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=lstm_units,
            dropout=dropout_rate,
            name="multihead_attention"
        )(x, x)
        x = Add(name="residual_attention")([x, attention_out])
        x = Dropout(dropout_rate, name="dropout_attention")(x)
    logits = TimeDistributed(
        Dense(num_labels, kernel_regularizer=l2(regularization)),
        name="time_dense"
    )(x)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    if crf_layer:
        crf = CRF(units=num_labels, dtype='float32')
        output = crf(logits)
        base_model = Model(inputs, output)
        model = ModelWithCRFLoss(base_model, sparse_target=False)
        model.compile(optimizer=optimizer)
    else:
        model = Model(inputs, logits)
        loss_fn = CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=["accuracy"])
    return model
