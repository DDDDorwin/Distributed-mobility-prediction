import os
import json
import pandas as pd
from matplotlib import pyplot as plt


# Function to create a JSON file with MSE values
def create_mse_json(filename, mse_values, epochs):
    if len(mse_values) != len(epochs):
        raise ValueError("mse_values and epochs must be the same length")
    mse_dict = {f"{epoch}": mse for mse, epoch in zip(mse_values, epochs)}
    with open(filename, "w") as f:
        json.dump(mse_dict, f)


def plot_mse_from_json(directory, saving_path=None):
    models = []
    print()
    if not len(os.listdir(directory)) > 0:
        print(f"No such directory: {directory}")
        return
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                data = json.load(f)
            temp_df = pd.DataFrame(list(data.values()), index=list(data.keys()), columns=[filename.split(".")[0]])
            models.append(temp_df)

    df = pd.concat(models, axis=1)
    df.plot()
    plt.title("MSE Loss of Models")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    if saving_path:
        plt.savefig(saving_path)
    plt.show()
