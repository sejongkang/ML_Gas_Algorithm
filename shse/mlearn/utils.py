import numpy as np
import torch

__all__ = ['save_model', 'load_model', 'value_to_onehot', 'onehot_to_value']


def save_model(model, file_name):
    return torch.save(model, file_name)


def load_model(file_name, map_location=None):
    return torch.load(file_name, map_location=map_location)


def value_to_onehot(values):
    labels = np.unique(values)

    encoding = np.array([labels == value for value in values], dtype=np.int)
    return encoding


def onehot_to_value(onehot_x, label=None):
    if label is None:
        label = np.shape(onehot_x)[-1]

    print(np.shape(onehot_x)[-1])


if __name__ == '__main__':
    x = np.arange(10)
    print(x)
    r = value_to_onehot(x)
    print(r)
    print(onehot_to_value(r))
