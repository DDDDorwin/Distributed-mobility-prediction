import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time
import argparse
from preprocessing import load_pickle, Paths, Keys
from models.models import OneDimensionalCNN, LSTM
from data.data import SequenceDataset, resize_input_data, make_test_set
from utils.plot import plot_test_graph, plot_loss

if __name__ == '__main__':
    # parse the input from the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='epochs for training, it will be 10 if not specified.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, 0.001 by default')
    parser.add_argument('--data', type=str, help='path of input data')
    parser.add_argument('--period', type=int, default=6, help='numbers of data points used for training: 6 as one '
                                                              'hour, 240 as one day')
    parser.add_argument('--output_size', type=int, default=1, help='numbers of prediction')
    parser.add_argument('--batch', type=int, default=1, help='batch size, 1 by default as it is time series prediction')

    args = parser.parse_args()
    data_path = args.data
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    output_size = args.output_size
    period = args.period  # numbers of data points used for training: 6 as one hour, 240 as one day

    # Set device: cpu, mps(macos), cuda
    device = (
        "cuda"
        if torch.cuda.is_available()
        # uncomment following 2 lines if you want to run
        # the training on Apple silicon with pytorch >= 2.0

        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load data
    raw_data = load_pickle(data_path)
    sum_data = raw_data.groupby(Keys.TIME_INTERVAL).sum()
    print(sum_data.head())

    # Apply MinMaxScaler normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(sum_data.values)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    norm_y = scaler_y.fit_transform(sum_data.values[:, -1].reshape(-1, 1))

    # Get subset of the ouput
    real_y = sum_data.values[:, -1]

    # make custom dataset
    resize_data = resize_input_data(norm_data, real_y, period, output_size)
    dataset = SequenceDataset(resize_data)

    # split data for training and testing
    train_size = int(len(sum_data) * 0.8)
    test_size = len(dataset) - train_size
    train_split_ratio = [train_size, test_size]

    train_set = Subset(dataset, range(train_size))
    test_set = Subset(dataset, range(train_size, len(dataset)))
    test_data = make_test_set(norm_data[train_size:], real_y[train_size:], period, output_size)
    # test_set = SequenceDataset(test_data)

    # Add dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    # Resize the input to fit the model
    # Set input size as one hour

    model = OneDimensionalCNN(period, output_size).double()
    lstm = LSTM(4, 10, 2, batch_first=True, batch_size=batch_size, nc=period).double()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Start training
    print("start training")
    losses = []
    for epoch in range(epochs):
        start_time = time.time()
        for batch, (seq, y_label) in enumerate(train_loader):
            seq, y_label = seq.to(device), y_label.to(device)
            # resize the label shape from (1, 1) to (1) so that it is the same shape with the input
            y_label = y_label

            # input shape: (batch_size, channel, series_length): (1, 1, -1)
            y_pred = lstm(seq)
            loss = loss_fn(y_label, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print(f'Epoch: {epoch + 1:2} Loss: {loss.item():10.8f}')
        print(f'\nDuration: {time.time() - start_time:.5f} seconds')

    # plot loss
    plot_loss(losses)

    # evaluation
    preds = []
    test_loss, correct = 0, 0
    running_mae, running_mse = 0.0, 0.0
    # set eval mode
    model.eval()
    # loop for sliding window
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = lstm(x)
            preds.append(pred)
            error = torch.abs(pred-y).sum().data
            squared_error = ((pred-y)*(pred-y)).sum().data
            running_mae += error
            running_mse += squared_error

    # accuracy evaluation with MSE method, the lower value the better result we get
    mae = running_mae / len(test_loader)
    mse = running_mse / len(test_loader)

    print(f"MAE value: {mae:.5f}, MSE value: {mse:.5f}")

    # reverse the normalization
    # true_predictions = scaler_y.inverse_transform(torch.cat(preds).numpy().reshape(-1, 1))
    # true_predictions = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
    true_predictions = (torch.cat(preds).numpy().reshape(-1, 1))

    plot_loss(losses)
    plot_test_graph(sum_data, true_predictions)