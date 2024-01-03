import numpy as np


def index_to_array(index, state_size):
    """
    :param index: 0-state_size
    :param state_size: a positive integer
    :return: a numpy array of length state_size
    """
    array = np.zeros(state_size)
    array[index] = 1
    array = np.reshape(array, [1, state_size])
    return array


def array_to_index(array, state_size):
    """
    :param array: a numpy array of length state_size
    :param state_size: a positive integer
    :return: an integer
    """
    index = array.tolist()[0].index(1)
    return index
