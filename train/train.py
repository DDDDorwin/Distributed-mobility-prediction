import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import time
from preprocessing import load_data
from models.CNN import resize_input_data, OneDimensionalCNN

if __name__ == '__main__':
    # Set device: cpu, mps(macos), cuda
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load data
    raw_data = load_data(parse_dates=True, input_dir="D:\Project_cs\data\\")
    sum_data = raw_data.groupby('Time_interval').sum()

    # Apply MinMaxScaler normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm_data = scaler.fit_transform(sum_data['Internet_traffic'].values.reshape(-1, 1))

    torch_data = torch.FloatTensor(norm_data).view(-1)
    train_size = int(len(sum_data) * 0.8)
    train_set = torch_data[:train_size]
    test_set = torch_data[train_size:]

    # Resize the input to fit the model
    # Set input size as one hour
    train_input = resize_input_data(train_set, 6)

    model = OneDimensionalCNN(1, 1)
    # set hyper parameters
    hyp = {
        'lr': 1e-4,
        'epochs': 10
    }
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr'])

    # Start training
    print("start training")

    for epoch in range(hyp['epochs']):
        start_time = time.time()
        for seq, y_train in train_input:
            seq, y_train = seq.to(device), y_train.to(device)
            optimizer.zero_grad()

            # input shape: (batch_size, channel, series_length): (1, 1, -1)
            y_pred = model(seq.reshape(1, 1, -1))
            loss = loss_fn(y_train, y_pred)
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
    for i in range(len(test_set) - 6):
        seq = torch.FloatTensor(test_set[i:i + 6])
        with torch.no_grad():
            pred = model(seq.reshape(1, 1, -1)).item()
            preds.append(pred)

    MSE = ((torch.pow((torch.FloatTensor(preds)-test_set[:-6]), 2)).sum()) / len(test_set)
    print(f"Accuracy: {MSE*100}%")

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