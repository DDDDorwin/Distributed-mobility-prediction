import torch
import torch.nn as nn


def resize_input_data(x, input_size):
    """
    resize the input data to fit the input shape of the network
    """
    output = []
    length = len(x)
    for i in range(length - input_size):
        window = x[i: i + input_size]
        pred = x[i+input_size: i+input_size + 1]

        output.append((window, pred))

    return output


class OneDimensionalCNN(nn.Module):
    def __init__(self, channel, input_size, nc):
        """
        One dimensional neural network
        One data point prediction using previous several data points (input_size)

        :param input_size: dimension of the input
        :param nc: numbers of the output (numbers of the labels)
        """
        super().__init__()
        self.channel = channel
        self.input_size = input_size
        self.nc = nc
        self.model = nn.Sequential(
            nn.Conv1d(channel, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, nc)
        )
    def forward(self, x):
        output = self.model(x)
        return output
