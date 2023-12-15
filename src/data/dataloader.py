from torch.utils.data import DataLoader, Subset


def get_data_loaders(data, batch_size):
    train_size = int(len(data[0]) * 0.6)
    eval_size = int(len(data[0]) * 0.8)
    test_size = len(data[0])

    train_set = Subset(data[0], range(train_size))
    eval_set = Subset(data[0], range(train_size, eval_size))
    test_set = Subset(data[0], range(eval_size, test_size))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, eval_loader, test_loader