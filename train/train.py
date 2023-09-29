import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time
import argparse
from preprocessing import load_data
from models.models import resize_input_data, OneDimensionalCNN

if __name__ == '__main__':
    # parse the input from the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='epochs for training, it will be 10 if not specified.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, 0.001 by default')
    parser.add_argument('--data', type=str, help='path of input data')
    parser.add_argument('--period', type=int, default=6, help='numbers of data points used for training: 6 as one '
                                                              'hour, 240 as one day')
    parser.add_argument('--batch', type=int, default=1, help='batch size, 1 by default as it is time series prediction')

    args = parser.parse_args()
    data_path = args.data
    lr = args.lr
    epochs = args.epochs
    period = args.period  # numbers of data points used for training: 6 as one hour, 240 as one day

    # Set device: cpu, mps(macos), cuda
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load data
    raw_data = load_data(parse_dates=True, input_dir=data_path)
    sum_data = raw_data.groupby('Time_interval').sum()

    # Apply MinMaxScaler normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(sum_data['Internet_traffic'].values.reshape(-1, 1))

    # split data for training and testing
    torch_data = torch.FloatTensor(norm_data).view(-1)
    train_size = int(len(sum_data) * 0.8)
    train_set = torch_data[:train_size]
    test_set = torch_data[train_size:]

    # Resize the input to fit the model
    # Set input size as one hour
    train_input = resize_input_data(train_set, period)

    model = OneDimensionalCNN(1, period, 1)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Start training
    print("start training")

    for epoch in range(epochs):
        start_time = time.time()
        for seq, y_label in train_input:
            seq, y_label = seq.to(device), y_label.to(device)
            # resize the label shape from (1, 1) to (1) so that it is the same shape with the input
            y_label = y_label.unsqueeze(1)

            # input shape: (batch_size, channel, series_length): (1, 1, -1)
            y_pred = model(seq.reshape(1, 1, -1))
            loss = loss_fn(y_label, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1:2} Loss: {loss.item():10.8f}')
        print(f'\nDuration: {time.time() - start_time:.5f} seconds')

    # evaluation
    preds = []
    test_loss, correct = 0, 0
    # set eval mode
    model.eval()
    # loop for sliding window
    for i in range(len(test_set) - period):
        seq = torch.FloatTensor(test_set[i:i + period])
        with torch.no_grad():
            pred = model(seq.reshape(1, 1, -1)).item()
            preds.append(pred)

    MSE = mean_squared_error(preds, test_set[:len(preds)])  # MSE
    print(f"Accuracy: {MSE:.5f}")

    # reverse the normalization
    true_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # compute accuracy
    # test set of true data
    test_true_data = sum_data['Internet_traffic'][len(sum_data) - len(true_predictions):]

    # plot the data
    plt.grid(True)
    plt.plot(sum_data.index[len(sum_data) - len(true_predictions):],
             sum_data['Internet_traffic'][len(sum_data) - len(true_predictions):])
    plt.plot(sum_data.index[len(sum_data) - len(true_predictions):], true_predictions)
    plt.show()
    plt.savefig("prediction_with_test_set.jpg")