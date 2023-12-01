import numpy as np
import wandb
import torch
import pandas as pd
from constants import Keys
from utils.util import inverse_normalization, normalization, load_sum_data

def plot_test_data(args, pred, test_loader):
    true_data = []
    for x, y in test_loader:
        true_data.append(y)
    true_data = torch.cat(true_data).numpy().reshape(-1, 1)
    pred = torch.cat(pred).numpy().reshape(-1, 1)
    df = pd.DataFrame(pred)
    df['ground_truth'] = true_data
    df[Keys.INDEX] = [i for i in range(len(pred))]
    df.columns = ['prediction', 'ground_truth', Keys.INDEX]
    # columns = [Keys.INTERNET, Keys.INDEX]
    # test_table = wandb.Table(data=pred)

    wandb.log({"prediction": df})
