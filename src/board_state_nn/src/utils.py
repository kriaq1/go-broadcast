import matplotlib.pyplot as plt
import numpy as np


def save_prediction(input, predict, save_name):
    np.savetxt(save_name, predict, fmt='%d')


def plot_prediction(input, predict, save_name=None):
    assert NotImplemented()
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Input')
    ax[0].imshow(input)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, format='jpg', dpi=1024)
    plt.close(fig)
