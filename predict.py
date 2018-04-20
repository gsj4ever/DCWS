# -*- coding: utf-8 -*-
# Author: Kevin
# Created Date: 2018-04-08

import h5py
import numpy as np
from keras.models import model_from_json
from keras_contrib.utils import save_load_utils
from keras_contrib.layers import CRF

SEQ_LABELS = ['S', 'B', 'M', 'E', 'X']

# with open('./data/test_chars.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#
# test_chars = []
# for l in lines:
#     l = l.strip()
#     test_chars.append(l.split(" "))
#
# f = h5py.File('./data/test_data.hdf5', 'r')
# test_X = f['test_X'][:]
# f.close()

with open('./data/model_architecture.json', 'r') as f:
    model = model_from_json(f.read(), custom_objects={'CRF': CRF})
save_load_utils.load_all_weights(model, './data/model_weights.hdf5')


# test_Y = model.predict(test_X)
test_chars = "平 时 鲜 于 露 面 媒 体 的 中 央 纪 委 副 书 记 ， 也 在 近 期 密 集 亮 相 。".split(" ")
x = np.random.random((1, 80)) * 5000

print(x)

test_Y = model.predict(x)

test_labels = np.argmax(test_Y, axis=-1)

print(test_Y)

# segments = []
# for i in range(len(test_chars)):
#     words = ''
#     for j in range(min(len(test_chars[i]), len(test_labels[i]))):
#         if SEQ_LABELS[test_labels[i][j]] in ['S', 'E', 'X']:
#             words += test_chars[i][j]
#             words += ' '
#         elif SEQ_LABELS[test_labels[i][j]] in ['B', 'M']:
#             words += test_chars[i][j]
#
#     segments.append(words.strip())
#
# for segs in segments:
#     print(segs)




