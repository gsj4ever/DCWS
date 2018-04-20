# -*- coding: utf-8 -*-
# Author: Kevin
# Created Date: 2018-04-06

import os
import codecs
import numpy as np
import h5py
import pickle
from keras.preprocessing import sequence, text
from keras.utils import to_categorical


SEQ_LABELS = ['S', 'B', 'M', 'E', 'X']  # Sequence labels, X for padding
MAX_LEN = 80
totalLine = 0
longLine = 0
chars = []
labels = []


def load_corpus(path, codec='utf-8'):
    """

    :param path:
    :param codec:
    :return:
    """
    lines = []
    for root, dirs, files in os.walk(path):
        for file in files:
            current_path = os.path.join(root, file)
            corpus_file = codecs.open(current_path, 'r', codec)
            lines.extend(corpus_file.readlines())
            corpus_file.close()
    return lines


def processSegWithTag(seg, char_collector, label_collector, isEnd):
    global totalLine
    global longLine
    nn = len(seg)
    while nn > 0 and seg[nn - 1] != '/':
        nn = nn - 1

    word = seg[:nn - 1].strip()

    if len(word) == 1:
        char_collector.append(word)
        label_collector.append(SEQ_LABELS.index('S'))
    else:
        char_collector.append(word[0])
        label_collector.append(SEQ_LABELS.index('B'))
        for c in word[1:len(word) - 1]:
            char_collector.append(c)
            label_collector.append(SEQ_LABELS.index('M'))
        char_collector.append(word[-1])
        label_collector.append(SEQ_LABELS.index('E'))

    label_line = label_collector.copy()

    char_line = u''
    if word == '。' or isEnd:
        if len(char_collector) > MAX_LEN:
            longLine += 1
        totalLine += 1
        for c in char_collector:
            if char_line:
                char_line = char_line + c
            else:
                char_line = c

        # TODO: if char_line is too short, do not append it to chars
        chars.append(char_line)
        labels.append(label_line)

        del char_collector[:]
        del label_collector[:]


def process_lines(lines):
    for line in lines:
        line = line.strip()
        seeLeftB = False
        start = 0
        char_collector = []
        label_collector = []
        if not line:
            continue
        else:
            line_len = len(line)
            try:
                for i in range(line_len):
                    if line[i] == ' ':
                        if not seeLeftB:
                            seg = line[start:i]
                            if seg.startswith('['):
                                segLen = len(seg)
                                while segLen > 0 and seg[segLen - 1] != ']':
                                    segLen = segLen - 1
                                seg = seg[1:segLen - 1]
                                ss = seg.split(' ')
                                for s in ss:
                                    processSegWithTag(s, char_collector, label_collector, False)
                            else:
                                processSegWithTag(seg, char_collector, label_collector, False)
                            start = i + 1
                    elif line[i] == '[':
                        seeLeftB = True
                    elif line[i] == ']':
                        seeLeftB = False
                if start < line_len:
                    seg = line[start:]
                    if seg.startswith('['):
                        segLen = len(seg)
                        while segLen > 0 and seg[segLen - 1] != ']':
                            segLen = segLen - 1
                        seg = seg[1:segLen - 1]
                        ss = seg.split(' ')
                        ns = len(ss)
                        for i in range(ns - 1):
                            processSegWithTag(ss[i], char_collector, label_collector, False)
                            processSegWithTag(ss[-1], char_collector, label_collector, True)
                    else:
                        processSegWithTag(seg, char_collector, label_collector, True)
            except Exception as e:
                pass

    print("Total lines: " + str(totalLine))
    print("Long lines: " + str(longLine))

    return chars, labels


if __name__ == '__main__':
    """Preprocess train files
    """
    # ROOT_PATH = 'D:/data/corpus/2015'
    ROOT_PATH = '/apps/py36venv/projects/LangLab/corpus/2014'
    lines = load_corpus(ROOT_PATH)

    chars, labels = process_lines(lines)

    tokenizer = text.Tokenizer(filters='', char_level=True, oov_token='UNK')
    tokenizer.fit_on_texts(chars)
    # TODO: deal with characters with lower frequency, which might less than 10 times or 5 in percentage
    tokens = tokenizer.texts_to_sequences(chars)
    dict_size = tokenizer.word_index.get('UNK')

    tokens_pad = sequence.pad_sequences(tokens, MAX_LEN, padding='post', truncating='post', value=0)
    labels_pad = sequence.pad_sequences(labels, MAX_LEN, padding='post', truncating='post', value=SEQ_LABELS.index('X'))

    label_cat = to_categorical(labels_pad, len(SEQ_LABELS))

    train_X = np.array(tokens_pad)
    train_Y = np.array(label_cat)

    # Save tokenizer for predicting or testing
    with open('./data/tokenizer', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Save training data sets, including useful training parameters
    file = h5py.File('./data/train_data.hdf5', 'w')
    file.create_dataset('train_X', data=train_X)
    file.create_dataset('train_Y', data=train_Y)
    file.create_dataset('params', data=(dict_size, MAX_LEN))
    file.close()

    print('Training data files processing finished!')
