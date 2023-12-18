import os
import sys
sys.path.append("D:\AAA\Projectcs\CODE\Project_CS_UserVsSpecific\src\gvslearning")

import torch
import wandb
import yaml

from gvslearning.data.dataloader import get_data_loaders
from gvslearning.test.test import test_main
from gvslearning.train.new_train import train_main
from gvslearning.utils.util import get_dataset
from gvslearning.utils.wandb_plot import plot_test_data


def run():
    print("****  Running gvslearning model ****")
    config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "hyp.yaml")
    with open(config_file, "r") as stream:
        args = yaml.safe_load(stream)

    wandb.init(
        project="Project_CS",
        config={
            "learning-rate": args["learning-rate"],
            "architecture": "LSTM",
            "dataset": "Milan",
            "epochs": args["epoch"],
            "batch": args["batch-size"],
        },
    )

    dataset, validate_dateset, test_dataset = get_dataset(args)
    datasets = [dataset, validate_dateset, test_dataset]
    # dataset = PickleDataset(train_size=args.period, test_size=args.output_size, max_saved_chunks=1)

    train_loader, eval_loader, test_loader = get_data_loaders(datasets, args["batch-size"])

    model = train_main(args, train_loader, eval_loader)

    # model = torch.load("./src/models/model/best.pt")
    print("test")

    test_pred = test_main(model, test_loader, args)
    plot_test_data(test_pred, test_loader)
    # plot_true_data(test_loader)

    wandb.finish()


if __name__ == "__main__":
    run()
