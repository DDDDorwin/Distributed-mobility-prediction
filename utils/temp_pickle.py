import os

import numpy as np
import pandas as pd
from data.data import Dataset
from utils.util import pickle_normalization
from typing import TypeVar
from constants import Paths, Keys, TableData
from os.path import join

def make_general_pickle():
    size = 0
    df = pd.DataFrame()
    DTYPES = {
        Keys.SQUARE_ID: "int16",
        Keys.TIME_INTERVAL: "int64",
        Keys.COUNTRY_CODE: "int8",
        Keys.SMS_IN: "float32",
        Keys.SMS_OUT: "float32",
        Keys.CALL_IN: "float32",
        Keys.CALL_OUT: "float32",
        Keys.INTERNET: "float32",
    }

    data_files = os.listdir(r"D:\project\python\project_cs\data\raw")
    input_files = [f
        for f in os.listdir(r"D:\project\python\project_cs\data\validate")
        if f.endswith(".txt")
    ]
    columns = ['square_id', 'time_interval', 'country_code', 'sms_in',
               'sms_out', 'call_in', 'call_out', 'internet_traffic']
    # Read the data and concat them
    df = pd.DataFrame()
    for file in input_files:
        read = pd.read_csv(r"D:\project\python\project_cs\data\validate/" + file, sep='\t', header=None, names=columns, parse_dates=True, dtype=DTYPES)
        # read = pd.read_csv(r"D:\project\python\project_cs\data\raw\/" + file, sep='\t', header=None, names=columns,
        #                    parse_dates=True)
        df = pd.concat([df, read], ignore_index=True)
        print("reading "+file)

    length = len(df)


    # df = df.fillna(0)
    df['start_time'] = pd.to_datetime(df[Keys.TIME_INTERVAL], unit='ms', utc=True).dt.tz_convert(
        'CET').dt.tz_localize(None)
    df = df.groupby([pd.Grouper(key='start_time', freq='10Min')]).sum()
    # df = df.groupby([Keys.SQUARE_ID, pd.Grouper(key='start_time', freq='10Min')]).agg()
    df = df.drop(Keys.TIME_INTERVAL, axis=1)
    df = df.drop(Keys.COUNTRY_CODE, axis=1)
    # ids.name = Keys.SQUARE_ID
    # ids.index = [i for i in range(len(ids))]
    normalized = pickle_normalization(df)
    # normalized[Keys.INDEX] = ids
    length = len(normalized)
    # Add corresponding indexes to rows
    normalized[Keys.INDEX] = [i for i in range(size, size + length)]

    # normalized = pd.concat([ids, normalized], axis=1)
    # normalized.insert(5, Keys.SQUARE_ID, ids)
    normalized = normalized.set_index(Keys.INDEX)
    # normalized[Keys.SQUARE_ID] = ids
    # normalized = pd.concat([ids, normalized], axis=1)
    # Fetches keys for dtypes, to use as names for headers
    normalized.columns = TableData.NORMALIZED_DTYPES.keys()
    # Make pickles with name: startIndex_endIndex.pkl
    normalized.to_pickle(r"D:\project\python\project_cs\data\validate.pkl")
    # Increment size
    size += length
    print("Successfully pickled")

def make_general_pickle():
    size = 0
    df = pd.DataFrame()
    DTYPES = {
        Keys.SQUARE_ID: "int16",
        Keys.TIME_INTERVAL: "int64",
        Keys.COUNTRY_CODE: "int8",
        Keys.SMS_IN: "float32",
        Keys.SMS_OUT: "float32",
        Keys.CALL_IN: "float32",
        Keys.CALL_OUT: "float32",
        Keys.INTERNET: "float32",
    }

    data_files = os.listdir(r"D:\project\python\project_cs\data\raw")
    input_files = [f
        for f in os.listdir(r"D:\project\python\project_cs\data\validate")
        if f.endswith(".txt")
    ]
    columns = ['square_id', 'time_interval', 'country_code', 'sms_in',
               'sms_out', 'call_in', 'call_out', 'internet_traffic']
    # Read the data and concat them
    df = pd.DataFrame()
    for file in input_files:
        read = pd.read_csv(r"D:\project\python\project_cs\data\validate/" + file, sep='\t', header=None, names=columns, parse_dates=True, dtype=DTYPES)
        # read = pd.read_csv(r"D:\project\python\project_cs\data\raw\/" + file, sep='\t', header=None, names=columns,
        #                    parse_dates=True)
        df = pd.concat([df, read], ignore_index=True)
        print("reading "+file)

    length = len(df)


    # df = df.fillna(0)
    df['start_time'] = pd.to_datetime(df[Keys.TIME_INTERVAL], unit='ms', utc=True).dt.tz_convert(
        'CET').dt.tz_localize(None)
    df = df.groupby([pd.Grouper(key='start_time', freq='10Min')]).sum()
    # df = df.groupby([Keys.SQUARE_ID, pd.Grouper(key='start_time', freq='10Min')]).agg()
    df = df.drop(Keys.TIME_INTERVAL, axis=1)
    df = df.drop(Keys.COUNTRY_CODE, axis=1)
    # ids.name = Keys.SQUARE_ID
    # ids.index = [i for i in range(len(ids))]
    normalized = pickle_normalization(df)
    # normalized[Keys.INDEX] = ids
    length = len(normalized)
    # Add corresponding indexes to rows
    normalized[Keys.INDEX] = [i for i in range(size, size + length)]

    # normalized = pd.concat([ids, normalized], axis=1)
    # normalized.insert(5, Keys.SQUARE_ID, ids)
    normalized = normalized.set_index(Keys.INDEX)
    # normalized[Keys.SQUARE_ID] = ids
    # normalized = pd.concat([ids, normalized], axis=1)
    # Fetches keys for dtypes, to use as names for headers
    normalized.columns = TableData.NORMALIZED_DTYPES.keys()
    # Make pickles with name: startIndex_endIndex.pkl
    normalized.to_pickle(r"D:\project\python\project_cs\data\validate.pkl")
    # Increment size
    size += length
    print("Successfully pickled")

def make_general_pickle():
    size = 0
    df = pd.DataFrame()
    DTYPES = {
        Keys.SQUARE_ID: "int16",
        Keys.TIME_INTERVAL: "int64",
        Keys.COUNTRY_CODE: "int8",
        Keys.SMS_IN: "float32",
        Keys.SMS_OUT: "float32",
        Keys.CALL_IN: "float32",
        Keys.CALL_OUT: "float32",
        Keys.INTERNET: "float32",
    }

    data_files = os.listdir(r"D:\project\python\project_cs\data\raw")
    input_files = [f
        for f in os.listdir(r"D:\project\python\project_cs\data\validate")
        if f.endswith(".txt")
    ]
    columns = ['square_id', 'time_interval', 'country_code', 'sms_in',
               'sms_out', 'call_in', 'call_out', 'internet_traffic']
    # Read the data and concat them
    df = pd.DataFrame()
    for file in input_files:
        read = pd.read_csv(r"D:\project\python\project_cs\data\validate/" + file, sep='\t', header=None, names=columns, parse_dates=True, dtype=DTYPES)
        # read = pd.read_csv(r"D:\project\python\project_cs\data\raw\/" + file, sep='\t', header=None, names=columns,
        #                    parse_dates=True)
        df = pd.concat([df, read], ignore_index=True)
        print("reading "+file)

    length = len(df)


    # df = df.fillna(0)
    df['start_time'] = pd.to_datetime(df[Keys.TIME_INTERVAL], unit='ms', utc=True).dt.tz_convert(
        'CET').dt.tz_localize(None)
    df = df.groupby([pd.Grouper(key='start_time', freq='10Min')]).sum()
    # df = df.groupby([Keys.SQUARE_ID, pd.Grouper(key='start_time', freq='10Min')]).agg()
    df = df.drop(Keys.TIME_INTERVAL, axis=1)
    df = df.drop(Keys.COUNTRY_CODE, axis=1)
    # ids.name = Keys.SQUARE_ID
    # ids.index = [i for i in range(len(ids))]
    normalized = pickle_normalization(df)
    # normalized[Keys.INDEX] = ids
    length = len(normalized)
    # Add corresponding indexes to rows
    normalized[Keys.INDEX] = [i for i in range(size, size + length)]

    # normalized = pd.concat([ids, normalized], axis=1)
    # normalized.insert(5, Keys.SQUARE_ID, ids)
    normalized = normalized.set_index(Keys.INDEX)
    # normalized[Keys.SQUARE_ID] = ids
    # normalized = pd.concat([ids, normalized], axis=1)
    # Fetches keys for dtypes, to use as names for headers
    normalized.columns = TableData.NORMALIZED_DTYPES.keys()
    # Make pickles with name: startIndex_endIndex.pkl
    normalized.to_pickle(r"D:\project\python\project_cs\data\validate.pkl")
    # Increment size
    size += length
    print("Successfully pickled")


if __name__=='__main__':
    make_general_pickle()