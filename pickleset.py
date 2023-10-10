import os 
import pandas as pd
from data.data import Dataset
from typing import TypeVar
from constants import *
from os.path import join


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class PickleDataset(Dataset[T_co]):
    is_loading = False
    loaded = [pd.DataFrame]
    size = 0

    #TODO: IMPLEMENT if needed
    def __init__(self, max_saved_chunks):
        None

    def __make_pickles__(self, destroy_old):
        '''Create pickle files for each raw data file, name = firstTimeStamp_lastTimeStamp.pkl'''
        if destroy_old:
            #Delete existing pickle files
            self.__del_db()
        #For all data files ending with .txt, .tsv or .csv, reformat them into pickles, adding corresponding headers in the process
        input_files = [join(Paths.RAW_DIR, f) for f in os.listdir(Paths.RAW_DIR) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
        for input_file in sorted(input_files):
            df = pd.read_csv(input_file, header=None, sep='\t', dtype=TableData.DTYPES)
            self.size += len(df)
            #Fetches keys for dtypes, to use as names for headers
            df.columns = TableData.DTYPES.keys()
            df.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (df[Keys.TIME_INTERVAL].iloc[1],df[Keys.TIME_INTERVAL].iloc[-1])))
            print("Successfully pickled %s" % (input_file))

    def __del_db(self):
        '''Deletes all pickles from pickles directory'''
        for file in os.listdir(Paths.PICKLE_DIR):
            if os.fsdecode(file).endswith(".pkl"):
                os.remove(join(Paths.PICKLE_DIR, file))
        self.size = 0
    
    #TODO: IMPLEMENT
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
    
    #TODO: IMPLEMENT
    def __get_saved_index__(self, index):
        for df in self.loaded:
            if(not df.empty and int(df[Keys.TIME_INTERVAL].iloc[-1]) >= index and int(df[Keys.TIME_INTERVAL].iloc[1]) <= index):
                return df
        return pd.DataFrame()
    
    #TODO: IMPLEMENT
    def __add_chunk_to_saved__(self):
        None



    def __fetch_chunk__(self, index):
        '''Returns a dataframe with headers, containing data from picklefile that contains the given index'''
        if index < 0:
            return None

        chunk = self.__get_saved_index__(index)
        #Check if dataframe is already loaded for requested timeframe
        if(len(chunk) > 0):
            print("Index already loaded, no fetch needed.")
            return chunk
        loaded = pd.DataFrame()
        print("Timeframe not loaded, fetching...")
        #Get missing pickle file in directory
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        for p in pickles:
            times = p.replace(".pkl",'').split("_", 1)
            #Check if file contains data in range
            if(int(times[1]) >= index and int(times[0]) <= index):
                #Append to return dataframe
                loaded = pd.read_pickle(join(Paths.PICKLE_DIR, p))
                print("Range contained in pickle: %s" % (p))

        #If nothing is being prefetched, save chunk
        self.__add_chunk_to_saved__(loaded)
        return loaded







    #TODO: IMPLEMENT
    def __len__(self):
        None

print(len(pd.DataFrame()))


pklst = PickleDataset()
#pklst.__make_pickles__(True)
print(pklst.size)
