from torch.utils.data import Dataset
import torch


# TODO: make a method for multi variant input


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
    x = x.to_numpy()
    y = y.to_numpy()
    for i in range(length - input_size):
        window = x[i : i + input_size, :]
        pred = y[i + input_size : i + input_size + prediction_size]
        output.append((window, pred))
        # print(i + input_size + prediction_size)

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
