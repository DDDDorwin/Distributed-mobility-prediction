import torch
import torch.nn as nn


class OneDimensionalCNN(nn.Module):
    def __init__(self, input_size, nc):
        """
        One dimensional neural network
        One data point prediction using previous several data points (input_size)

        :param input_size: dimension of the input
        :param nc: numbers of the output (numbers of the labels)
        """
        super().__init__()
        # self.channel = channel
        self.input_size = input_size
        self.nc = nc
        self.model = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, nc)
        )

    def forward(self, x):
        output = self.model(x)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, batch_size, nc):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.batch_size = batch_size

        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.fc2 = nn.Linear(in_features=input_size, out_features=1)

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(self.num_layers * hidden_size, nc)
        self.dropout = nn.Dropout(0.2)

    # def forward(self, x):
    #     h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64)
    #     c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64)
    #     lstm_output, hn = self.lstm(x, (h_0, c_0))
    #     # lstm_output, _ = self.lstm(x, None)
    #     # ac = self.activation(lstm_output)
    #     output = self.fc1(lstm_output[:, -1, :])
    #     hn = hn[0].view(self.batch_size, self.hidden_size)
    #     output = self.fc1(hn)
    #     # output = self.activation(output)
    #     # output = self.fc2(output)
    #
    #     return output

    def forward(self, x):
        batch_size = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # lstm layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        x = h_n.permute(1, 0, 2).reshape(batch_size, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]
