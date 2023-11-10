from torch.utils.data import Dataset
import torch


def resize_input_data(x, y, input_size, prediction_size):
    """
    resize the input data to fit the input shape of the network
    using 'internet_traffic' as output
    :param x training input
    :param y training output
    :param input_size Integer defining how many steps to look back
    :param prediction_size Integer defining how many steps forward to predict
    """

    output = []
    length = len(x)
    for i in range(length - input_size):
        window = x[i: i + input_size, :]
        pred = y[i + input_size: i + input_size + prediction_size]

        output.append((window, pred))

    return output


def make_test_set(x, y, input_size, prediction_size):
    """
    make the test set with an input_size given
    :param x: test data
    :param y: test label
    :param input_size: the size of the sliding window
    :param prediction_size: should be not useful
    :return: list of (window, output)
    """
    output = []
    idx = 0
    length = int(len(x) / input_size)
    for i in range(length):
        window = x[idx: idx + input_size, :]
        pred = y[idx: idx + input_size]
        idx += 6
        output.append((window, pred))

    return output


class SequenceDataset(Dataset):
    """
    Custom dataset class for creating our own dataset with time series data points
    """

    def __init__(self, data):
        """
        :param data dataset input (list)
        """
        self.data = data

    def __len__(self):
        """
        returns the number of samples in our dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        loads and returns a sample from the dataset at the given index idx
        :param idx index of an object in the list

        returns sample[0] the look back window, sample[1] the prediction window
        """
        sample = self.data[idx]
        return torch.from_numpy(sample[0]), torch.from_numpy(sample[1])
        # return torch.Tensor(sample[0]), torch.Tensor(sample[1])
