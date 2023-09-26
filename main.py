import time

from utils.preprocessing import load_data, input_data
from models.Conv import CNNnetwork
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    raw_data = load_data('../cdr/')
    sum_data = raw_data.groupby('start_time').sum()

    # Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm_data = scaler.fit_transform(sum_data['Internet_traffic'].values.reshape(-1, 1))

    torch_data = torch.FloatTensor(norm_data).view(-1)
    train_size = int(len(sum_data) * 0.8)
    train_set = torch_data[:train_size]
    test_set = torch_data[train_size:]

    train_input = input_data(train_set, 6)

    model = CNNnetwork()

    # loss function
    loss_func = nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100

    # set train mode
    model.train()

    start_time = time.time()

    print("start training")
    # training
    for epoch in range(epochs):
        for seq, y_train in train_input:
            optimizer.zero_grad()

            # conv1d dimension (batch_size, channel, series_length): (1, 1, -1)
            y_pred = model(seq.reshape(1, 1, -1))

            loss = loss_func(y_pred, y_train)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1:2} Loss: {loss.item():10.8f}')
    print(f'\nDuration: {time.time() - start_time:.5f} seconds')


    # evaluation
    preds = []
    # set eval mode
    model.eval()
    # loop for sliding window
    for i in range(len(test_set)-6):
        seq = torch.FloatTensor(torch_data[i:i+6])
        with torch.no_grad():
            preds.append(model(seq.reshape(1, 1, -1)).item())

    # reverse the normalization
    true_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # plot the data
    plt.grid(True)
    plt.plot(sum_data.index[len(sum_data)-len(true_predictions):], sum_data['Internet_traffic'][len(sum_data)-len(true_predictions):])
    plt.plot(sum_data.index[len(sum_data)-len(true_predictions):], true_predictions)
    plt.show()

