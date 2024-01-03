# Unit test
import os
import shutil
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
            "metric-test/json/mse/model2.json",
            mse_values_model2,
            epochs_model2,
        )
        msevis.plot_mse_from_json(
            "metric-test/json/mse",
            "metric-test/image/mse.png",
        )

    def test_more_models(self):
        mse_values_model1 = [0.1, 0.08, 0.05, 0.03, 0.01]
        mse_values_model2 = [0.2, 0.18, 0.15, 0.13, 0.11]
        epochs_model1 = [1, 2, 3, 4, 5]
        epochs_model2 = [1, 2, 3, 4, 5]
        msevis.create_mse_json(
            "metric-test/json/mse/model1.json",
            mse_values_model1,
            epochs_model1,
        )
        msevis.create_mse_json(
            "metric-test/json/mse/model2.json",
            mse_values_model2,
            epochs_model2,
        )
        msevis.plot_mse_from_json(
            "metric-test/json/mse/",
            "metric-test/image/mse.png",
        )

    def test_no_jsons(self):
        msevis.plot_mse_from_json(
            "metric-test/json/mse/",
            "metric-test/image/mse.png",
        )

    def test_draw_table(self):
        mse_values_model1 = [0.1, 0.08, 0.05, 0.03, 0.01]
        mse_values_model2 = [0.2, 0.18, 0.15, 0.13, 0.11]
        mse_values_model3 = [0.4, 0.58, 0.35, 0.03, 0.09]
        mse_values_model4 = [0.8, 0.78, 0.05, 0.3, 0.31]
        epochs_model1 = [1, 2, 3, 4, 5]
        epochs_model2 = [1, 2, 3, 4, 5]
        epochs_model3 = [1, 2, 3, 4, 5]
        epochs_model4 = [1, 2, 3, 4, 5]
        msevis.create_mse_json(
            "metric-test/json/mse/model1.json",
            mse_values_model1,
            epochs_model1,
        )
        msevis.create_mse_json(
            "metric-test/json/mse/model2.json",
            mse_values_model2,
            epochs_model2,
        )
        msevis.create_mse_json(
            "metric-test/json/mse/model3.json",
            mse_values_model3,
            epochs_model3,
        )
        msevis.create_mse_json(
            "metric-test/json/mse/model4.json",
            mse_values_model4,
            epochs_model4,
        )
        msevis.make_table_from_json(
            "metric-test/json/mse/",
            "metric-test/image/table_mse.png",
        )

    def tearDown(self):
        if os.path.exists("metric-test"):
            shutil.rmtree("metric-test")


if __name__ == "__main__":
    unittest.main()
