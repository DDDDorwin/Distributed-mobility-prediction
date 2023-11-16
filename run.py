import argparse
import wandb

from utils.util import get_dataset
from data.dataloader import get_data_loaders
from data.data import resize_input_data, SequenceDataset
from pickleset import PickleDataset
from new_train import train_main
from test import test_main
from eval import eval_main


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='epochs for training, it will be 10 if not specified.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, 0.001 by default')
    parser.add_argument('--data', type=str, help='path of input data')
    parser.add_argument('--device', type=str, default='cpu', help='path of input data')
    parser.add_argument('--period', type=int, default=6, help='numbers of data points used for training: 6 as one '
                                                              'hour, 240 as one day')
    parser.add_argument('--output_size', type=int, default=1, help='numbers of prediction')
    parser.add_argument('--batch', type=int, default=1, help='batch size, 1 by default as it is time series prediction')

    args = parser.parse_args()

    wandb.init(
        project="Project_CS",
        config={
            "learning-rate": args.lr,
            "architecture": "LSTM",
            "dataset": "Milan",
            "epochs": args.epoch,
            "batch": args.batch,
        }
    )

    dataset = get_dataset(args)
    # dataset = PickleDataset(train_size=args.period, test_size=args.output_size, max_saved_chunks=1)

    train_loader, eval_loader, test_loader = get_data_loaders(dataset, args.batch)

    model = train_main(args, train_loader, eval_loader)

    print("test")
    test_pred = test_main(model, test_loader, args)

    wandb.finish()


if __name__ == '__main__':
    run()
