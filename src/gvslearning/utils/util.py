from gvslearning.utils.constants import Keys
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset

from gvslearning.data.data import SequenceDataset, resize_input_data


def load_sum_data(data_path):
    raw_data = pd.read_pickle(data_path, compression=None)
    sum_data = raw_data.groupby(Keys.TIME_INTERVAL).sum()
    print(sum_data.head())

    return sum_data


def load_data(data_path):
    raw_data = pd.read_pickle(data_path, compression=None)
    print(raw_data.head())

    return raw_data


def normalization(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(data.values)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    norm_y = scaler_y.fit_transform(data.values[:, -1].reshape(-1, 1))

    return norm_data, norm_y, scaler_y


def pickle_normalization(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(data.values)
    df = pd.DataFrame(norm_data)

    return df


def inverse_normalization(data, scaler):
    return scaler.inverse_transform(data)


def split_dataset(data):
    train_size = int(len(data) * 0.6)
    eval_size = int(len(data) * 0.8) - train_size
    test_size = len(data) - eval_size

    train_set = Subset(data, range(train_size))
    eval_set = Subset(data, range(eval_size))
    test_set = Subset(data, range(test_size))

    return train_set, eval_set, test_set


def get_dataset(args):
    train_data = load_data(args["data"]["train-data"])
    validate_data = load_data(args["data"]["eval-data"])
    test_data = load_data(args["data"]["test-data"])
    # norm_data, norm_label, _ = normalization(data)

    resize_data = resize_input_data(train_data, train_data[Keys.INTERNET], args["period"], args["output-size"])
    validate_resize_data = resize_input_data(
        validate_data, validate_data[Keys.INTERNET], args["period"], args["output-size"]
    )
    
    test_resize_data = resize_input_data(test_data, test_data[Keys.INTERNET], args["period"], args["output-size"])

    # data_train_wide = torch.tensor(resize_data).reshape((len(resize_data), args["period"], 5, 1))
    # data_train_test_wide = resize_data.reshape((len(resize_data), args["period"], 5, 1))
    #
    # data_validate_wide = torch.tensor(validate_resize_data).reshape((len(validate_resize_data), args["period"], 5, 1))
    # data_validate_test_wide = resize_data.reshape((len(validate_resize_data), args["period"], 5, 1))
    #
    # data_test_wide = torch.tensor(test_resize_data).reshape((len(test_resize_data), args["period"], 5, 1))
    # data_test_wide = resize_data.reshape((len(test_resize_data), args["period"], 5, 1))

    dataset = SequenceDataset(resize_data)
    validate_dataset = SequenceDataset(validate_resize_data)
    test_dataset = SequenceDataset(test_resize_data)

    return dataset, validate_dataset, test_dataset


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    model = torch.load(path)
    return model
