import os
import json
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import table


class MseVisualization:
    # Function to create a JSON file with MSE values
    @staticmethod
    def create_mse_json(filename, mse_values, epochs):
        if len(mse_values) != len(epochs):
            raise ValueError("mse_values and epochs must be the same length")
        mse_dict = {f"{epoch}": mse for mse, epoch in zip(mse_values, epochs)}
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(mse_dict, f)

    @staticmethod
    def plot_mse_from_json(directory, saving_path=None):
        table_data = []
        models = []
        if not os.path.exists(directory):
            print(f"{directory} is not a directory")
            return
        if not len(os.listdir(directory)) > 0:
            print(f"No such directory: {directory}")
            return
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), "r") as f:
                    data = json.load(f)
                table_data.append(data)
                temp_df = pd.DataFrame(list(data.values()), index=list(data.keys()), columns=[filename.split(".")[0]])
                models.append(temp_df)
        df = pd.concat(models, axis=1)
        df.plot()
        plt.title("MSE Loss of Models")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        if not os.path.exists(os.path.dirname(saving_path)):
            os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        plt.savefig(saving_path)

    @staticmethod
    def make_table_from_json(directory, table_saving_path=None):
        models = []
        if not os.path.exists(directory):
            print(f"{directory} is not a directory")
            return
        if not len(os.listdir(directory)) > 0:
            print(f"No such directory: {directory}")
            return
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), "r") as f:
                    data = json.load(f)
                # Select the highest MSE value from the JSON file
                max_mse = max(data.values())
                temp_df = pd.DataFrame([max_mse], index=[filename.split(".")[0]], columns=["MSE"])
                models.append(temp_df)
        df = pd.concat(models)
        # Save table as a separate figure
        ax = plt.subplot(frame_on=False)  # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        table(ax, df, loc="center")
        if not os.path.exists(os.path.dirname(table_saving_path)):
            os.makedirs(os.path.dirname(table_saving_path), exist_ok=True)
        plt.savefig(table_saving_path)
