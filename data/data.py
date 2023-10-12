from torch.utils.data import Dataset
import torch
import numpy as np

def resize_input_data(x, input_size, prediction_size):
    """
    resize the input data to fit the input shape of the network
    :param x training input
    :param input_size Integer defining how many steps to look back
    :param prediction_size Integer defining how many steps forward to predict
    """

    output = []
    data_x = []
    data_y = []
    length = len(x)
    for i in range(length - input_size):
        window = x[i: i + input_size]
        pred = x[i + input_size: i + input_size + prediction_size]

        data_x.append(window)
        data_y.append(pred)

        output.append((np.array(window), np.array(pred)))

    return np.array(data_x), np.array(data_y)


class SequenceDataset(Dataset):
    """
    Custom dataset class for creating our own dataset with time series data points
    """

    def __init__(self, data_x, data_y):
        """
        :param data dataset input (list)
        """
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __len__(self):
        """
        returns the number of samples in our dataset
        """
        return self.len

    def __getitem__(self, idx):
        """
        loads and returns a sample from the dataset at the given index idx
        :param idx index of an object in the list

        returns sample[0] the look back window, sample[1] the prediction window
        """
        # sample = self.data[idx]
        return torch.Tensor(self.data_x[idx]), torch.Tensor(self.data_y[idx])
