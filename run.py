import argparse

from utils.util import get_dataset
from data.dataloader import get_data_loaders
from data.data import resize_input_data, SequenceDataset

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='epochs for training, it will be 10 if not specified.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, 0.001 by default')
    parser.add_argument('--data', type=str, help='path of input data')
    parser.add_argument('--period', type=int, default=6, help='numbers of data points used for training: 6 as one '
                                                              'hour, 240 as one day')
    parser.add_argument('--output_size', type=int, default=1, help='numbers of prediction')
    parser.add_argument('--batch', type=int, default=1, help='batch size, 1 by default as it is time series prediction')

    args = parser.parse_args()

    dataset = get_dataset(args)

    train_loader, eval_loader, test_loader = get_data_loaders(dataset, args.batch)

    print("hello")




