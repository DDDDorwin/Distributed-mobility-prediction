import wandb
import torch
from constants import Keys
from utils.util import inverse_normalization, normalization, load_sum_data

def plot_test_data(args, pred):
    data = load_sum_data(args.data)
    norm_data, norm_label, scaler = normalization(data)
    true_predictions = scaler.inverse_transform(torch.cat(pred).numpy().reshape(-1, 1))
    columns = ['True Data', 'Pred Data']
    time_steps = data.index[len(data) - len(pred):]
    xs = [i for i in range(len(data) - len(pred), len(data))]
    ys = [data[Keys.INTERNET][len(data) - len(pred):], [i for i in true_predictions]]
    wandb.log({"Test Set Data": wandb.plot.line_series(
        xs=xs,
        ys=ys,
        keys=columns,
        title="Test Set Comparison")})
