import wandb
import torch
from constants import Keys
import pandas as pd
from utils.util import inverse_normalization, normalization, load_sum_data

def plot_test_data(args, pred):
    pred = torch.cat(pred).numpy().reshape(-1, 1)
    df = pd.DataFrame(pred)
    df[Keys.INDEX] = [i for i in range(len(pred))]
    df.columns = [Keys.INTERNET, Keys.INDEX]
    # columns = [Keys.INTERNET, Keys.INDEX]
    # test_table = wandb.Table(data=pred)

    wandb.log({"prediction": df})
