from threading import Thread
from os.path import join
from constants import *
import pandas as pd
import numpy as np
import os
import sys
import csv
import pickle
import queue

class __Data:
    is_loading = False
    loaded_chunk = pd.DataFrame()

def fetch_chunk(from_time, to_time):
    '''Returns a dataframe with headers, containing rows from from_time to to_time'''
    #Return None if input is faulty
    if(to_time < from_time or from_time < 0 or to_time < 0):
        return None
    #Check if dataframe is already loaded for requested timeframe
    if(__loaded_contains_range(from_time, to_time)):
        print("Timeframe already loaded, returning...")
        return __Data.loaded_chunk
    print("Timeframe not loaded, fetching...")
    #Get all pickles in directory
    pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
    to_load = []
    for p in pickles:
        times = p.replace(".pkl",'').split("_", 1)
        #Check if file contains data in range
        if(int(times[1]) >= from_time and int(times[0]) <= to_time):
            #Append to return dataframe
            to_load.append(pd.read_pickle(join(Paths.PICKLE_DIR, p)))

            print("Range contained in pickle: %s" % (p))
    chnk = pd.concat(to_load)

    #If nothing is being prefetched, save chunk
    if(not __Data.is_loading):
        __Data.loaded_chunk = chnk
    return chnk

def preload_chunk(from_time, to_time):
    #If nothing is being prefetched, start prefetch
    if(not __Data.is_loading):
        #Set status to loading as fast as possible
        __Data.is_loading = True
        prefetch_thread = Thread(target = __prefetch, args = (from_time, to_time, ))
        prefetch_thread.start()
        print("Prefetch started.")
    print("Prefetch failed to start.")

def fetch_preload():
    #Wait untill prefetch is available
    while(__Data.is_loading):
        pass
    return __Data.loaded_chunk

def is_ready():
    '''Returns true if chunk is ready for fetching'''
    return not __Data.is_loading

def __loaded_contains_range(from_time, to_time):
    if(not __Data.loaded_chunk.empty and int(__Data.loaded_chunk[Keys.TIME_INTERVAL].iloc[-1]) >= to_time and int(__Data.loaded_chunk[Keys.TIME_INTERVAL].iloc[1]) <= from_time):
        print(int(__Data.loaded_chunk[Keys.TIME_INTERVAL].iloc[-1]))
        return True
    return False

def __prefetch(from_time, to_time):
    __Data.loaded_chunk = fetch_chunk(from_time, to_time)
    __Data.is_loading = False

def __make_pickles():
    '''Create pickle files for each raw data file, name = firstTimeStamp_lastTimeStamp.pkl'''
    #For all data files ending with .txt, .tsv or .csv, reformat them into pickles, adding corresponding headers in the process
    input_files = [join(Paths.RAW_DIR, f) for f in os.listdir(Paths.RAW_DIR) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
    for input_file in sorted(input_files):
        df = pd.read_csv(input_file, header=None, sep='\t', dtype=TableData.DTYPES)
        #Fetches keys for dtypes, to use as names for headers
        df.columns = TableData.DTYPES.keys()
        df.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (df[Keys.TIME_INTERVAL].iloc[1],df[Keys.TIME_INTERVAL].iloc[-1])))
        print("Successfully pickled %s" % (input_file))

def __del_db():
    '''Deletes all pickles from pickles directory'''
    for file in os.listdir(Paths.PICKLE_DIR):
        if os.fsdecode(file).endswith(".pkl"):
            os.remove(join(Paths.PICKLE_DIR, file))

def __prep_db():
    '''Recreates database from scratch using all raw .txt files'''
    __del_db()
    __make_pickles()


#__prep_db()
print(preload_chunk(1383260400000, 1383605400000))
print(fetch_preload())
print(preload_chunk(1383260400000, 1383605400000))
print(fetch_preload())
print(preload_chunk(1383260400000, 1383605400000))
print(fetch_preload())
print(preload_chunk(1383260400000, 1383605400000))
print(fetch_preload())





