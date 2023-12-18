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
        self.feature_number = 0 # number of features used for training
        self.input_size = input_size
        self.nc = nc
        self.model = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nc)
        )

    def forward(self, x):

        output = self.model(x)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, batch_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.batch_size = batch_size
        self.embedding_size = embedding_size

        self.embedding_sid = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_sin = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_sout = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_cin = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_cout = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_internet = nn.Linear(in_features=input_size, out_features=embedding_size)

        self.lstm = nn.LSTM(6, hidden_size, num_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.fc2 = nn.Linear(in_features=input_size, out_features=1)

    def forward(self, x):
        embedding_output = []
        # sid = self.embedding_sin(x[:, :, 0])
        # sin = self.embedding_sin(x[:, :, 1])
        # sout = self.embedding_sin(x[:, :, 2])
        # cin = self.embedding_sin(x[:, :, 3])
        # cout = self.embedding_sin(x[:, :, 4])
        # internet = self.embedding_sin(x[:, :, 5])
        #
        # concat = torch.cat((sid, sin, sout, cin, cout), dim=1)


        # out = self.embedding(x.view(-1))
        h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64)
        c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64)


        lstm_output, _ = self.lstm(x, (h_0, c_0))
        # ac = self.activation(lstm_output)
        output = self.fc1(lstm_output[:, -1, :])

        return output


class lstm_embedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, batch_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.batch_size = batch_size
        self.embedding_size = embedding_size

        self.embedding_sid = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_sin = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_sout = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_cin = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_cout = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.embedding_internet = nn.Linear(in_features=input_size, out_features=embedding_size)

        self.lstm = nn.LSTM(6*self.embedding_size, hidden_size, num_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.fc2 = nn.Linear(in_features=input_size, out_features=1)

    def forward(self, x):
        embedding_output = []
        sid = self.embedding_sin(x[:, :, 0])
        sin = self.embedding_sin(x[:, :, 1])
        sout = self.embedding_sin(x[:, :, 2])
        cin = self.embedding_sin(x[:, :, 3])
        cout = self.embedding_sin(x[:, :, 4])
        internet = self.embedding_sin(x[:, :, 5])

        concat = torch.cat((sid, sin, sout, cin, cout, internet), dim=1)


        # out = self.embedding(x.view(-1))
        h_0 = torch.randn(self.num_layers, self.hidden_size, dtype=torch.float64)
        c_0 = torch.randn(self.num_layers, self.hidden_size, dtype=torch.float64)


        lstm_output, _ = self.lstm(concat, (h_0, c_0))
        # ac = self.activation(lstm_output)
        output = self.fc1(lstm_output)

        return output


class BasicConv2D(nn.Module):
    def __init__(self, n_filters=24, fsize=2, window_size=6, n_features=5):
        super(BasicConv2D, self).__init__()

        # Conv2D layer
        self.conv1 = nn.Conv2d(in_channels=n_features, out_channels=n_filters, kernel_size=(2, fsize), padding='same')
        # Calculate the flattened size after Conv2D
        self.flattened_size = n_filters * window_size * 1
        # Dense layers
        self.fc1 = nn.Linear(self.flattened_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        x = torch.relu(self.conv1(x))
        x = x.view(-1, self.flattened_size)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x