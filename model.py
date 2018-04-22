# -*- coding: utf-8 -*-
# Author: Kevin
# Created Date: 2018-04-03

import h5py
import pickle
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, TimeDistributed, Dropout
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras_contrib.utils import save_load_utils


"""
Global Parameters in training
"""
EMBEDDING_SIZE = 100
HIDDEN_UNITS = 128
DROPOUT_RATE = 0.5
BATCH_SIZE = 1024
EPOCH_NUM = 50


def build_model(x, y, vocab_size, max_len):
    """Build up and train a bi-directional LSTM + CRF model, saving model architecture and weights, as well as history
    :param x:
    :param y:
    :param vocab_size:
    :param max_len:
    :return:
    """

    # TODO: read from an existing Word2Vec model, to enhance embedding performance

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=EMBEDDING_SIZE, input_length=max_len, mask_zero=True))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))

    # TODO: consider to add a CNN layer to get higher accuracies

    model.add(TimeDistributed(Dense(HIDDEN_UNITS, activation='relu')))
    crf = CRF(5)  # CAUTION!!! sparse_target: True for index, False for one-hot
    model.add(crf)
    model.summary()

    model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])

    checkpointer = ModelCheckpoint(filepath='./data/weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor="val_loss", patience=2)
    terminator = TerminateOnNaN()
    history = model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCH_NUM, validation_split=0.1, verbose=1,
                        callbacks=[checkpointer, stopper, terminator])

    # Save model architecture and weights
    with open('./data/model_architecture.json', 'w') as f:
        f.write(model.to_json())
    save_load_utils.save_all_weights(model, './data/model_weights.hdf5')

    with open('./data/history', 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    with h5py.File('./data/train_data.hdf5', 'r') as file:
        train_X = file['train_X'][:]
        train_Y = file['train_Y'][:]
        (dict_size, MAX_LEN) = file['params']
    print("Training data sets loaded!")

    build_model(train_X, train_Y, dict_size, MAX_LEN)
    print("Model building and training finished!")
