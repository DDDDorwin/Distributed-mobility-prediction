import wandb
import torch
import pandas as pd
from gvslearning.utils.constants import Keys
from gvslearning.utils.util import get_date


def plot_test_data(pred, test_loader, args):
    true_data = []
    for x, y in test_loader:
        true_data.append(y)
    true_data = torch.cat(true_data).numpy().reshape(-1, 1)
    pred = torch.cat(pred).numpy().reshape(-1, 1)
    df = pd.DataFrame(pred)
    date = get_date(args["data"]["test-date"])
    # df['ground_truth'] = true_data
    df[Keys.INDEX] = date.values[:856]
    df.columns = ["data", "date"]
    # columns = [Keys.INTERNET, Keys.INDEX]
    # test_table = wandb.Table(data=pred)

    wandb.log({"data": df})


def plot_true_data(test_loader, args):
    true_data = []
    for x, y in test_loader:
        true_data.append(y)
    true_data = torch.cat(true_data).numpy().reshape(-1, 1)
    df = pd.DataFrame(true_data)
    date = get_date(args["data"]["test-date"])
    # df['ground_truth'] = true_data
    df[Keys.INDEX] = date.values[:856]
    df.columns = ["data", "date"]
    # columns = [Keys.INTERNET, Keys.INDEX]
    # test_table = wandb.Table(data=pred)

    wandb.log({"data": df})
