import os.path
import time
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from gvslearning.models.models import lstm_embedding
from gvslearning.eval.eval import eval_main
from gvslearning.utils.util import save_model


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

        if batch % 1000 == 0:
            print("batch:" + str(batch))
            print(f"\nDuration: {time.time() - start_time:.5f} seconds")
            start_time = time.time()
    wandb.log({"train loss": running_loss})


def train_main(args, train_loader, eval_loader):
    lrs = []
    losses = []
    if not os.path.exists("./src/models/model"):
        os.mkdir("./src/models/model")
    best_val_loss = float("inf")

    # define model
    # model = LSTM(args.period, 16, 2, batch_first=True, batch_size=args["batch-size"], embedding_size=16).double()
    model = lstm_embedding(
        args["period"], 128, 2, batch_first=False, batch_size=args["batch-size"], embedding_size=16
    ).double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args["learning-rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=args["epoch"])

    print("start training")
    for epoch in range(args["epoch"]):
        train(model, train_loader, optimizer, criterion, args["batch-size"], args["device"])
        scheduler.step()
        wandb.log({"learning rate": optimizer.param_groups[0]["lr"]})
        lrs.append(optimizer.param_groups[0]["lr"])

        # validation
        val_loss = eval_main(model, eval_loader, criterion, args["device"])

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "./src/models/model/best.pt")

    save_model(model, "./src/models/model/last.pt")

    return model
