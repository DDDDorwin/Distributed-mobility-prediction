import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from models.models import BasicConv2D
from data.dataloader import get_data_loaders
from eval import eval_main
from test import test_main
from utils.util import save_model, load_model


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

    wandb.log({"train loss": running_loss})
    print(f'\nDuration: {time.time() - start_time:.5f} seconds')


def train_main(args, train_loader, eval_loader):
    lrs = []
    losses = []
    best_val_loss = float('inf')
    model = BasicConv2D(24,2,6,6).double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)

    print("start training")
    for epoch in range(args.epoch):
        train(model, train_loader, optimizer, criterion, args.batch, args.device)
        scheduler.step()
        wandb.log({"learning rate": optimizer.param_groups[0]["lr"]})
        lrs.append(optimizer.param_groups[0]["lr"])

        # validation
        val_loss = eval_main(model, eval_loader, criterion, args.device)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "./models/model/best.pt")

    save_model(model, "./models/model/last.pt")

    return model