import argparse
import torch
import wandb
import yaml
import os

from src.train.new_train import train_main
from src.utils.util import get_dataset
from src.utils.wandb_plot import plot_test_data
from src.data.dataloader import get_data_loaders
from src.test.test import test_main


def run():
    with open("./src/main/hyp.yaml", 'r') as stream:
        args = yaml.safe_load(stream)

    wandb.init(
        project="Project_CS",
        config={
            "learning-rate": args["learning-rate"],
            "architecture": "LSTM",
            "dataset": "Milan",
            "epochs": args["epoch"],
            "batch": args["batch-size"],
        }
    )

    dataset, validate_dateset, test_dataset = get_dataset(args)
    datasets = [dataset, validate_dateset, test_dataset]
    # dataset = PickleDataset(train_size=args.period, test_size=args.output_size, max_saved_chunks=1)

    train_loader, eval_loader, test_loader = get_data_loaders(datasets, args["batch-size"])

    model = train_main(args, train_loader, eval_loader)

    model = torch.load("./src/models/model/best.pt")
    print("test")

    test_pred = test_main(model, test_loader, args)
    plot_test_data(test_pred, test_loader)
    # plot_true_data(test_loader)


    wandb.finish()


if __name__ == '__main__':
    run()
