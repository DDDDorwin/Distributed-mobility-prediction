# Unit test
import os
import unittest

from visualization.mse import create_mse_json, plot_mse_from_json


class TestMSEPlot(unittest.TestCase):
    def test_only_one_model(self):
        mse_values_model2 = [0.2, 0.18, 0.15, 0.13, 0.11]
        epochs_model2 = [1, 2, 3, 4, 5]
        create_mse_json("../assets/jsons/evaluation_metrics/mse/model2.json", mse_values_model2, epochs_model2)
        plot_mse_from_json("../assets/jsons/evaluation_metrics/mse/", "../assets/images/evaluation_metrics/mse.png")

    def test_more_models(self):
        mse_values_model1 = [0.1, 0.08, 0.05, 0.03, 0.01]
        mse_values_model2 = [0.2, 0.18, 0.15, 0.13, 0.11]
        epochs_model1 = [1, 2, 3, 4, 5]
        epochs_model2 = [1, 2, 3, 4, 5]
        create_mse_json("../assets/jsons/evaluation_metrics/mse/model1.json", mse_values_model1, epochs_model1)
        create_mse_json("../assets/jsons/evaluation_metrics/mse/model2.json", mse_values_model2, epochs_model2)
        plot_mse_from_json("../assets/jsons/evaluation_metrics/mse/", "../assets/images/evaluation_metrics/mse.png")

    def test_no_jsons(self):
        plot_mse_from_json("../assets/jsons/evaluation_metrics/mse/")

    def tearDown(self):
        if os.path.exists("../assets/jsons/evaluation_metrics/mse/model1.json"):
            os.remove("../assets/jsons/evaluation_metrics/mse/model1.json")
        if os.path.exists("../assets/jsons/evaluation_metrics/mse/model2.json"):
            os.remove("../assets/jsons/evaluation_metrics/mse/model2.json")
        if os.path.exists("../assets/images/evaluation_metrics/mse.png"):
            os.remove("../assets/images/evaluation_metrics/mse.png")


if __name__ == "__main__":
    unittest.main()
