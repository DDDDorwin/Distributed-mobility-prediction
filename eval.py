import torch
import wandb

def validate(model, val_loader, criterion, device):
    model.eval()
    current_val_loss = 0.0
    correct = 0
    running_mae, running_mse = 0.0, 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y = y.reshape(x.shape[0], 1).double()
            pred = model(x)
            loss = criterion(y, pred)

            error = torch.abs(pred - y).sum().data
            squared_error = ((pred - y) * (pred - y)).sum().data
            running_mae += error
            running_mse += squared_error

    mae = running_mae / len(val_loader)
    mse = running_mse / len(val_loader)

    print(f"MAE value: {mae:.5f}, MSE value: {mse:.5f}")
    print(f'Loss: {loss.item():10.8f}')
    wandb.log({"eval loss": loss.item()})
    return loss


def eval_main(model, val_loader, criterion, device):
    loss = validate(model, val_loader, criterion, device)
    return loss
