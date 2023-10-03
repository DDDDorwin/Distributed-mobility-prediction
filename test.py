import pandas as pd
import os
from os.path import join
import sys
import csv
from threading import Thread
import pickle
import numpy as np

pickle_path = "data/"
pickle_name = "temp_pickle"
raw_data_path = "data/raw_test/"


__DTYPES = {
    'sq_id': 'int16',
    'time': 'int64',
    'cc': 'int8',
    'sms_in': 'float64',
    'sms_out': 'float64',
    'call_in': 'float64',
    'call_out': 'float64',
    'internet': 'float64'
}

__is_loading = False
__loaded_chunk = pd.DataFrame()
__chunk_size = 2000

__n_pkl_files = 10

def __prefetch(first_elem_time):
    #Prefetch data, once done isLoading = false
    __isLoading = False

def preload_chunk(first_elem_time):
    __isLoading = True
    prefetch_thread = Thread(target = __prefetch, args = (first_elem_time, ))
    prefetch_thread.start()

def fetch_preload():
    #Loop/ wait in case chunk is not available yet
    None

def is_ready():
    return not __is_loading

def test_pkl():
    df = pd.read_pickle(join(pickle_path,pickle_name))
    print(df)

def __merge_and_pickle_csvs():
    #Fetch paths of all txt, tsv and csv's into a list
    input_files = [join(raw_data_path, f) for f in os.listdir(raw_data_path) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
    li = []
    #Merge all data from txt, tsv and csv's into a list
    for input_file in sorted(input_files):
        li.append(pd.read_csv(input_file, header=None, sep='\t', dtype=__DTYPES))
    #Make list into a dataframe
    df = pd.concat(li, axis=0, ignore_index=True)
    #Add column headers and sort by time
    df.columns = __DTYPES.keys()
    df.sort_values(by="time")
    #Split dataframe into n_chunks and pickle each chunk
    chunks = np.array_split(df, __n_pkl_files)
    i = 0
    for chunk in chunks:
        df_ch = pd.DataFrame(chunk)
        df_ch.to_pickle(join(pickle_path, "%s%s.pkl" % (pickle_name,str(i))))
        i = i + 1
        

    #Convert and save dataframe to pickle
   # df.to_pickle(join(pickle_path,pickle_name,".pkl"))

#Used only for creation of database, do not call normally
def __prep_db():
    #if no merged file exists, merge csvs
    #load into pickle file
    None

def __del_db():
    for file in os.listdir(pickle_path):
        if os.fsdecode(file).endswith(".pkl"):
            os.remove(join(pickle_path, file))

__del_db()
__merge_and_pickle_csvs()
#test_pkl()