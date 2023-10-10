import os 
import pandas as pd
from data.data import Dataset
from typing import TypeVar
from constants import *
from os.path import join


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class PickleDataset(Dataset[T_co]):
    #TODO: IMPLEMENT
    def __init__(self):
        None

    def __make_pickles__(self):
        '''Create pickle files for each raw data file, name = firstTimeStamp_lastTimeStamp.pkl'''
        #For all data files ending with .txt, .tsv or .csv, reformat them into pickles, adding corresponding headers in the process
        input_files = [join(Paths.RAW_DIR, f) for f in os.listdir(Paths.RAW_DIR) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
        for input_file in sorted(input_files):
            df = pd.read_csv(input_file, header=None, sep='\t', dtype=TableData.DTYPES)
            #Fetches keys for dtypes, to use as names for headers
            df.columns = TableData.DTYPES.keys()
            df.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (df[Keys.TIME_INTERVAL].iloc[1],df[Keys.TIME_INTERVAL].iloc[-1])))
            print("Successfully pickled %s" % (input_file))

    #TODO: IMPLEMENT
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
    
    #TODO: IMPLEMENT
    def __len__(self):
        None

