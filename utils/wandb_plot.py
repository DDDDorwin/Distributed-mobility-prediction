import wandb
import torch
import pandas as pd
from constants import Keys
from utils.util import inverse_normalization, normalization, load_sum_data

def plot_test_data(args, pred):
    pred = torch.cat(pred).numpy().reshape(-1, 1)
    df = pd.DataFrame(pred)
    df[Keys.INDEX] = [i for i in range(len(pred))]
    # columns = [Keys.INTERNET, Keys.INDEX]
    # test_table = wandb.Table(data=pred)

    wandb.log({"prediction": df})
