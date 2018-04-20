# # -*- coding: utf-8 -*-
# # Author: Kevin
# # Created Date: 2018-04-06

import pickle
import matplotlib.pyplot as plt


def plot_history(history_path):
    with open(history_path, 'rb') as f:
        hist = pickle.load(f)

    iters = range(len(hist['loss']))
    plt.figure()
    plt.title('Training and ')
    # acc
    plt.subplot(211)
    plt.plot(iters, hist['acc'], 'r.-', label='training acc')
    plt.plot(iters, hist['val_acc'], 'b.-', label='validation acc')
    plt.grid(True)
    plt.ylabel('Accuracy ')
    plt.legend(loc="lower right")
    # loss
    plt.subplot(212)
    plt.plot(iters, hist['loss'], 'g.-', label='training loss')
    plt.plot(iters, hist['val_loss'], 'k.-', label='validation loss')
    plt.grid(True)
    plt.xlabel('epoch #')
    plt.ylabel('Loss')
    plt.legend(loc="right")
    plt.show()


if __name__ == '__main__':
    plot_history('./data/history')
