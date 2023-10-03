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
    '''Returns true if preloaded chunk is ready for fetching'''
    return not __is_loading

def __ferment_pickles():
    '''Create pickle files for each raw data file, name = firstTimeStamp_lastTimeStamp.pkl'''
    #For all data files ending with .txt, .tsv or .csv, reformat them into pickles, adding corresponding headers in the process
    input_files = [join(Paths.RAW_DIR, f) for f in os.listdir(Paths.RAW_DIR) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
    for input_file in sorted(input_files):
        df = pd.read_csv(input_file, header=None, sep='\t', dtype=TableData.DTYPES)
        #Fetches keys for dtypes, to use as names for headers
        df.columns = TableData.DTYPES.keys()
        df.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (df[Keys.TIME_INTERVAL].iloc[1],df[Keys.TIME_INTERVAL].iloc[-1])))

def __del_db():
    '''Deletes all pickles from pickles directory'''
    for file in os.listdir(Paths.PICKLE_DIR):
        if os.fsdecode(file).endswith(".pkl"):
            os.remove(join(Paths.PICKLE_DIR, file))

def __prep_db():
    '''Recreates database from scratch using all raw .txt files'''
    __del_db()
    __ferment_pickles()


__prep_db()