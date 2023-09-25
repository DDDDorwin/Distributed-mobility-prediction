import pandas as pd
import os
import sys


def load_data(data_path):
    data_files = os.listdir(data_path)

    if '.DS_Store' in data_files:
        data_files.remove('.DS_Store')

    columns = ['Square_id', 'Time_interval', 'Country_code', 'SMS_in',
               'SMS_out', 'Call_in', 'Call_out', 'Internet_traffic']

    raw = pd.DataFrame()
    for file in data_files:
        read = pd.read_csv(data_path + file, sep='\t', header=None, names=columns, parse_dates=True)
        raw = pd.concat([raw, read], ignore_index=True)

    # Convert time interval to date format
    raw['start_time'] = pd.to_datetime(raw.Time_interval, unit='ms', utc=True).dt.tz_convert('CET').dt.tz_localize(None)

    return raw


def input_data(seq, window_size):
    output = []
    length = len(seq)
    for i in range(length-window_size):
        window = seq[i: i+window_size]
        pred = seq[i+window_size: i+window_size+1]

        output.append((window, pred))

    return output