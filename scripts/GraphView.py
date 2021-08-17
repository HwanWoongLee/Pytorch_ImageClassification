import matplotlib.pyplot as plt
import numpy as np


def draw_graph(epoch, train_loss, val_loss, color='r'):
    plt.plot(epoch, train_loss, color, label='train loss')
    plt.plot(epoch, val_loss, 'gold', label='val loss', linestyle='--')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')

    plt.ylim([0, np.max(train_loss)])

    plt.savefig('train_graph.png')
    plt.show()

