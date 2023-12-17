from datetime import timedelta

import folium
import pandas as pd
from folium.plugins import HeatMapWithTime
from pandas import DataFrame
from shapely.geometry import Polygon

from gvslearning.utils.constants import Keys


class GeoHeatMapClass:
    def __init__(self, df: DataFrame, geo_json_data):
        self.df: DataFrame = df
        self.geo_json_data = geo_json_data

    def add_coordinate(self, row):
        """Work with data given, expects a dataframe"""
        polygon = Polygon(self.geo_json_data.features[row[Keys.SQUARE_ID] - 1].geometry.coordinates[0])
        # Compute the centroid of the polygon
        centroid = polygon.centroid
        return [centroid.y, centroid.x]

    def assign_coordinate_to_grid_id(self):
        print("assigning coordinates to GridID")
        self.df[Keys.COORDINATE] = self.df.apply(self.add_coordinate, axis=1)

    def add_hour_to_time(self):
        """Add 1 hour to the time interval, to align with the timezone for Milan"""
        self.df[Keys.TIME_INTERVAL] = pd.to_datetime(
            self.df[Keys.TIME_INTERVAL], format="%Y-%m-%d %H:%M:%S"
        ) + timedelta(hours=1)
        print("Adding 1 hour to the time interval")

    def group_data_by_date(self):
        """Extract the date from the time_interval column, and group the rows by column and date"""
        self.df[Keys.DATE] = self.df[Keys.TIME_INTERVAL].dt.date
        self.df = (
            self.df.groupby([Keys.SQUARE_ID, Keys.DATE])
            .agg({Keys.CELL_TRAFFIC: "sum", Keys.COORDINATE: "first"})
            .sort_values(by=Keys.DATE)
            .reset_index()
        )

    def time_series_heatmap(self, weight_column, time_index_column):
        """Creating the heatmap that is going to be displayed on the map"""
        """ With time series in mind """
        print("Making time series heatmap")
        unique_dates = self.df[time_index_column].unique()
        heatmap = folium.Map(location=[45.4642, 9.1900], zoom_start=12, tiles="cartodbpositron")
        # gradient = {0.2: '#0000FF60', 0.4: '#00800060', 0.6: '#FFFF0060', 1: '#FF000060'}
        data_for_heatmap_with_time = []
        date_index = []
        for date in unique_dates:
            date_data = self.df.loc[self.df[time_index_column] == date]
            date_index.append(date.strftime("%Y-%m-%d"))
            data_for_heatmap_with_time.append(
                [
                    [row[Keys.COORDINATE][0], row[Keys.COORDINATE][1], row[weight_column]]
                    for index, row in date_data.iterrows()
                ]
            )
        HeatMapWithTime(
            data_for_heatmap_with_time, index=date_index, min_opacity=0.1, max_opacity=0.9, use_local_extrema=True
        ).add_to(heatmap)
        return heatmap

    def generate_heat_map(self, feature, time_index_column=Keys.TIME_INTERVAL):
        print("Generating heatmap... 1/4")
        self.df = self.assign_coordinate_to_grid_id()
        print("Generating heatmap... 2/4")
        # might not be necessary
        self.df = self.add_hour_to_time()
        print("Generating heatmap... 3/4")
        self.df = self.group_data_by_date()
        print("Generating heatmap... 4/4")
        # Weight column solely depends on the feature we want to showcase here.
        # Time index column also depends on what we are using, by default it is time interval
        return self.time_series_heatmap(feature, time_index_column)
