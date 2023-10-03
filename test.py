import pandas as pd
import os
from os.path import join
import sys
import csv
from threading import Thread
import pickle
import numpy as np
from constants import *

__is_loading = False
__loaded_chunk = pd.DataFrame()

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



def __merge_and_pickle_csvs():
    #Fetch paths of all txt, tsv and csv's into a list
    input_files = [join(Paths.RAW_DIR, f) for f in os.listdir(Paths.RAW_DIR) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
    li = []
    #Merge all data from txt, tsv and csv's into a list
    for input_file in sorted(input_files):
        li.append(pd.read_csv(input_file, header=None, sep='\t', dtype=TableData.DTYPES))
    #Make list into a dataframe
    df = pd.concat(li, axis=0, ignore_index=True)
    #Add column headers and sort by time
    df.columns = TableData.DTYPES.keys()
    df.sort_values(by='time')
    #Split dataframe into n_chunks and pickle each chunk
    chunks = np.array_split(df, __n_pkl_files)
    for chunk in chunks:
        df_ch = pd.DataFrame(chunk)
        #Save pkl with name corresponding to start and end timestep in file
        df_ch.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (df_ch['time'].iloc[1],df_ch['time'].iloc[-1])))

#Used only for creation of database, do not call normally
def __prep_db():
    #if no merged file exists, merge csvs
    #load into pickle file
    None

def __del_db():
    for file in os.listdir(Paths.PICKLE_DIR):
        if os.fsdecode(file).endswith(".pkl"):
            os.remove(join(Paths.PICKLE_DIR, file))

__del_db()
__merge_and_pickle_csvs()
