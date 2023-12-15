import wandb
import torch
import pandas as pd
from src.utils.constants import Keys


def plot_test_data(pred, test_loader):
    true_data = []
    for x, y in test_loader:
        true_data.append(y)
    true_data = torch.cat(true_data).numpy().reshape(-1, 1)
    pred = torch.cat(pred).numpy().reshape(-1, 1)
    df = pd.DataFrame(pred)
    # df['ground_truth'] = true_data
    df[Keys.INDEX] = [i for i in range(len(pred))]
    df.columns = ['data', Keys.INDEX]
    # columns = [Keys.INTERNET, Keys.INDEX]
    # test_table = wandb.Table(data=pred)

    wandb.log({"data": df})

def plot_true_data(test_loader):
    true_data = []
    for x, y in test_loader:
        true_data.append(y)
    true_data = torch.cat(true_data).numpy().reshape(-1, 1)
    df = pd.DataFrame(true_data)
    df[Keys.INDEX] = [i for i in range(len(true_data))]
    df.columns = ['data', Keys.INDEX]
    # columns = [Keys.INTERNET, Keys.INDEX]
    # test_table = wandb.Table(data=pred)

    wandb.log({"data": df})
