# -*- coding: utf-8 -*-
# Author: Kevin
# Created Date: 2018-04-08

import pickle
import codecs
import h5py
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras_contrib.utils import save_load_utils
from keras_contrib.layers import CRF

SEQ_LABELS = ['S', 'B', 'M', 'E', 'X']
with h5py.File('./data/train_data.hdf5', 'r') as f:
    (dict_size, MAX_LEN) = f['params']

test_file = codecs.open("./data/test.txt", 'r', 'utf-8')
lines = test_file.readlines()
test_file.close()

test_sentences = []
for line in lines:
    line = line.strip()
    sents = line.split('。')
    if len(sents) == 1:
        test_sentences.extend(sents)
    else:
        for s in sents:
            if s != '':
                test_sentences.append(s.strip()+'。')
print(test_sentences)

# Reconstruct trained tokenizer
with open('./data/tokenizer', 'rb') as f:
    tokenizer = pickle.load(f)
# Load built model architecture and trained weights
with open('./data/model_architecture.json', 'r') as f:
    model = model_from_json(f.read(), custom_objects={'CRF': CRF})
save_load_utils.load_all_weights(model, './data/model_weights.hdf5')

test_tokens = tokenizer.texts_to_sequences(test_sentences)
test_tokens_pad = sequence.pad_sequences(test_tokens, maxlen=MAX_LEN, padding='post', truncating='post', value=0)
test_X = np.array(test_tokens_pad)
test_Y = model.predict(test_X)

test_chars = [list(i) for i in test_sentences]
test_labels = np.argmax(test_Y, axis=-1)

segments = []
for i in range(len(test_chars)):
    words = ''
    for j in range(min(len(test_chars[i]), len(test_labels[i]))):
        if SEQ_LABELS[test_labels[i][j]] in ['S', 'E', 'X']:
            words += test_chars[i][j]
            words += ' '
        elif SEQ_LABELS[test_labels[i][j]] in ['B', 'M']:
            words += test_chars[i][j]

    segments.append(words.strip())

for segs in segments:
    print(segs)
