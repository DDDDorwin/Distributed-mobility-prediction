import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from models.models import LSTM
from data.dataloader import get_data_loaders

def train(model, train_loader, optimizer, criterion, batch_size, device):
    model.train(True)
    running_loss = 0.0
    start_time = time.time()
    for batch, (seq, y_label) in enumerate(train_loader):
        seq, y_label = seq.to(device), y_label.to(device)
        # resize the label shape from (1, 1) to (1) so that it is the same shape with the input
        y_label = y_label.reshape(batch_size, 1).double()
        seq = seq.double()

        # input shape: (batch_size, channel, series_length): (1, 1, -1)
        y_pred = model(seq)
        loss = criterion(y_label, y_pred)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(args):
    lrs = []
    model = LSTM(5, 30, 2, batch_first=True, batch_size=args.batch_size).double()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train