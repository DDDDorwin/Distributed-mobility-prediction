import torch
import wandb


def test(model, test_loader, device, preds):
    model.eval()

    test_loss = 0.0
    correct = 0
    running_mae, running_mse = 0.0, 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y = y.reshape(x.shape[0], 1).double()
            pred = model(x)

            error = torch.abs(pred - y).sum().data
            squared_error = ((pred - y) * (pred - y)).sum().data
            running_mae += error
            running_mse += squared_error
            preds.append(pred)

    mae = running_mae / len(test_loader)
    mse = running_mse / len(test_loader)

    print(f"MAE value: {mae:.5f}, MSE value: {mse:.5f}")
    wandb.log({"MSE value": mse})


def test_main(model, test_loader, args):
    preds = []
    test(model, test_loader, args["device"], preds)

    return preds
