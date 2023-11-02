import unittest

from pandas import DataFrame
import pandas as pd

from constants import Keys
from src.visualization.GeoHeatMap import GeoHeatMapClass
from pandas import Timestamp
from datetime import datetime, date


class TestGeoHeatMapClass(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.df = DataFrame(
            {
                "square_id": [1, 2, 3, 4, 5],
                "time_interval": [
                    "2022-01-01 00:00:00",
                    "2022-01-01 01:00:00",
                    "2022-01-01 02:00:00",
                    "2022-01-02 00:00:00",
                    "2022-01-02 01:00:00",
                ],
                "cell_traffic": [10, 20, 30, 40, 50],
            }
        )

    def test_add_coordinate(self):
        # Create an instance of the GeoHeatMapClass
        geo_heatmap = GeoHeatMapClass(self.df)

        # Test the add_coordinate method with a sample row
        row = {"square_id": 1}
        # Center of the polygon of square_id 1
        expected_result = [45.3577435, 9.012990999999998]
        self.assertEqual(geo_heatmap.add_coordinate(row), expected_result)

    def test_add_hour_to_time(self):
        # Create an instance of the GeoHeatMapClass
        geo_heatmap = GeoHeatMapClass(self.df)

        # Test the add_hour_to_time method
        geo_heatmap.add_hour_to_time()
        expected_result = [
            Timestamp("2022-01-01 01:00:00"),
            Timestamp("2022-01-01 02:00:00"),
            Timestamp("2022-01-01 03:00:00"),
            Timestamp("2022-01-02 01:00:00"),
            Timestamp("2022-01-02 02:00:00"),
        ]
        self.assertListEqual(list(geo_heatmap.df["time_interval"]), expected_result)

    # Test case 1: Test if the date column is created correctly
    def test_group_data_by_date_create_date_column(self):
        # Create an instance of the GeoHeatMapClass
        geo_heatmap = GeoHeatMapClass(self.df)

        # Create a sample dataframe
        data = {
            "time_interval": [datetime(2022, 1, 1, 0, 0), datetime(2022, 1, 1, 1, 0), datetime(2022, 1, 2, 0, 0)],
            "square_id": [1, 1, 2],
            "cell_traffic": [10, 20, 30],
            "coordinate": ["1.0,2.0", "1.0,2.0", "3.0,4.0"],
        }
        df = pd.DataFrame(data)

        # Call the group_data_by_date method
        geo_heatmap.df = df
        geo_heatmap.group_data_by_date()

        # Check if the date column is created correctly
        assert list(geo_heatmap.df["date"]) == [date(2022, 1, 1), date(2022, 1, 2)]

    # Test case 2: Test if the rows are grouped by square ID and date correctly
    def test_group_data_by_date_group_by_square_id_and_date(self):
        # Create an instance of the GeoHeatMapClass
        geo_heatmap = GeoHeatMapClass(self.df)

        # Create a sample dataframe
        data = {
            "time_interval": [datetime(2022, 1, 1, 0, 0), datetime(2022, 1, 1, 1, 0), datetime(2022, 1, 2, 0, 0)],
            "square_id": [1, 1, 2],
            "cell_traffic": [10, 20, 30],
            "coordinate": ["1.0,2.0", "1.0,2.0", "3.0,4.0"],
        }
        df = pd.DataFrame(data)

        # Call the group_data_by_date method
        geo_heatmap.df = df
        geo_heatmap.group_data_by_date()

        # Check if the rows are grouped by square ID and date correctly
        assert list(geo_heatmap.df.groupby([Keys.SQUARE_ID, Keys.DATE]).groups.keys()) == [
            (1, date(2022, 1, 1)),
            (2, date(2022, 1, 2)),
        ]

    # Test case 3: Test if the cell traffic is summed correctly
    def test_group_data_by_date_sum_cell_traffic(self):
        # Create an instance of the GeoHeatMapClass
        geo_heatmap = GeoHeatMapClass(self.df)

        # Create a sample dataframe
        data = {
            "time_interval": [datetime(2022, 1, 1, 0, 0), datetime(2022, 1, 1, 1, 0), datetime(2022, 1, 2, 0, 0)],
            "square_id": [1, 1, 2],
            "cell_traffic": [10, 20, 30],
            "coordinate": ["1.0,2.0", "1.0,2.0", "3.0,4.0"],
        }
        df = pd.DataFrame(data)

        # Call the group_data_by_date method
        geo_heatmap.df = df
        geo_heatmap.group_data_by_date()

        # Check if the cell traffic is summed correctly
        assert list(
            geo_heatmap.df.groupby([Keys.SQUARE_ID, Keys.DATE]).agg({Keys.CELL_TRAFFIC: "sum"})[Keys.CELL_TRAFFIC]
        ) == [30, 30]

    # Test case 4: Test if the coordinate is taken from the first row correctly
    def test_group_data_by_date_take_coordinate_from_first_row(self):
        # Create an instance of the GeoHeatMapClass
        geo_heatmap = GeoHeatMapClass(self.df)

        # Create a sample dataframe
        data = {
            "time_interval": [datetime(2022, 1, 1, 0, 0), datetime(2022, 1, 1, 1, 0), datetime(2022, 1, 2, 0, 0)],
            "square_id": [1, 1, 2],
            "cell_traffic": [10, 20, 30],
            "coordinate": ["1.0,2.0", "1.0,2.0", "3.0,4.0"],
        }
        df = pd.DataFrame(data)

        # Call the group_data_by_date method
        geo_heatmap.df = df
        geo_heatmap.group_data_by_date()

        # Check if the coordinate is taken from the first row correctly
        assert list(
            geo_heatmap.df.groupby([Keys.SQUARE_ID, Keys.DATE]).agg({Keys.COORDINATE: "first"})[Keys.COORDINATE]
        ) == ["1.0,2.0", "3.0,4.0"]

    # def test_time_series_heatmap(self):
    #     # Create an instance of the GeoHeatMapClass
    #     geo_heatmap = GeoHeatMapClass(self.df)
    #     data = {'time_interval': [datetime(2022, 1, 1, 0, 0), datetime(2022, 1, 1, 1, 0),
    #                               datetime(2022, 1, 2, 0, 0)],
    #             'square_id': [1, 1, 2],
    #             'cell_traffic': [10, 20, 30],
    #             'coordinate': ['1.0,2.0', '1.0,2.0', '3.0,4.0']}
    #     df = pd.DataFrame(data)
    #     df = df[Keys.TIME_INTERVAL].dt.date
    #     geo_heatmap.df = df
    #
    #     # Test the time_series_heatmap method
    #     heatmap = geo_heatmap.time_series_heatmap(Keys.CELL_TRAFFIC, Keys.DATE)
    #     self.assertIsInstance(heatmap, folium.Map)
    #


if __name__ == "__main__":
    unittest.main()
