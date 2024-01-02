# Unit test
import os
import unittest

from gvslearning.visualization.mse import MseVisualization as msevis


class TestMSEPlot(unittest.TestCase):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_only_one_model(self):
        mse_values_model2 = [0.2, 0.18, 0.15, 0.13, 0.11]
        epochs_model2 = [1, 2, 3, 4, 5]
        msevis.create_mse_json(
            os.path.join(self.rootPath, "assets/jsons/evaluation_metrics/mse/model2.json"),
            mse_values_model2,
            epochs_model2,
        )
        msevis.plot_mse_from_json(
            os.path.join(self.rootPath, "assets/jsons/evaluation_metrics/mse/"),
            os.path.join(self.rootPath, "assets/images/evaluation_metrics/mse.png"),
        )

    def test_more_models(self):
        mse_values_model1 = [0.1, 0.08, 0.05, 0.03, 0.01]
        mse_values_model2 = [0.2, 0.18, 0.15, 0.13, 0.11]
        epochs_model1 = [1, 2, 3, 4, 5]
        epochs_model2 = [1, 2, 3, 4, 5]
        msevis.create_mse_json(
            os.path.join(self.rootPath, "assets/jsons/evaluation_metrics/mse/model1.json"),
            mse_values_model1,
            epochs_model1,
        )
        msevis.create_mse_json(
            os.path.join(self.rootPath, "assets/jsons/evaluation_metrics/mse/model2.json"),
            mse_values_model2,
            epochs_model2,
        )
        msevis.plot_mse_from_json(
            os.path.join(self.rootPath, "assets/jsons/evaluation_metrics/mse/"),
            os.path.join(self.rootPath, "assets/images/evaluation_metrics/mse.png"),
        )

    def test_no_jsons(self):
        msevis.plot_mse_from_json(os.path.join(self.rootPath, "assets/jsons/evaluation_metrics/mse/"))

    def tearDown(self):
        if os.path.exists("../assets/jsons/evaluation_metrics/mse/model1.json"):
            os.remove("../assets/jsons/evaluation_metrics/mse/model1.json")
        if os.path.exists("../assets/jsons/evaluation_metrics/mse/model2.json"):
            os.remove("../assets/jsons/evaluation_metrics/mse/model2.json")
        if os.path.exists("../assets/images/evaluation_metrics/mse.png"):
            os.remove("../assets/images/evaluation_metrics/mse.png")


if __name__ == "__main__":
    unittest.main()
