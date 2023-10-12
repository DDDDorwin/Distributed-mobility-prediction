import os 
import pandas as pd
import queue
from data.data import Dataset
from typing import TypeVar
from constants import *
from os.path import join


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class PickleDataset(Dataset):
    is_loading = False


    def __init__(self, max_saved_chunks):
        self.__get_size__()
        if max_saved_chunks < 1:
            max_saved_chunks = 1
        self.max_chunks = max_saved_chunks
        self.__loaded_chunks = [pd.DataFrame] * max_saved_chunks
        self.__loaded_chunks_next = 0



        #FIFO queue for saving chunks, queue size = max_saved_chunks
        self.loaded_chunkz = queue.Queue(max_saved_chunks)
        print("%s pickled rows of data" % self.size)

#SIZE RELATED FUNCTIONS
    def __update_size__(self, size):
        f = open("SIZE_DATA.txt", "w+")
        f.write('%d' % size)
        self.size = size

    def __get_size__(self):
        if(not os.stat("SIZE_DATA.txt").st_size == 0):       
            f = open("SIZE_DATA.txt", "r")
            self.size = int(f.readline())
        else:
            self.__update_size__(0)

    def __len__(self):
        return self.size

#CREATION/DELETION OF PICKLES
    def __make_pickles__(self, destroy_old):
        '''Create pickle files for each raw data file, name = firstTimeStamp_lastTimeStamp.pkl'''
        if destroy_old:
            #Delete existing pickle files
            self.__del_db__()
        #For all data files ending with .txt, .tsv or .csv, reformat them into pickles, adding corresponding headers in the process
        input_files = [join(Paths.RAW_DIR, f) for f in os.listdir(Paths.RAW_DIR) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
        size = 0
        for input_file in sorted(input_files):
            df = pd.read_csv(input_file, header=None, sep='\t', dtype=TableData.DTYPES)
            size += len(df)
            #Fetches keys for dtypes, to use as names for headers
            df.columns = TableData.DTYPES.keys()
            df.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (df[Keys.TIME_INTERVAL].iloc[1],df[Keys.TIME_INTERVAL].iloc[-1])))
            print("Successfully pickled %s" % (input_file))
        self.__update_size__(size)

    def __del_db__(self):
        '''Deletes all pickles from pickles directory'''
        for file in os.listdir(Paths.PICKLE_DIR):
            if os.fsdecode(file).endswith(".pkl"):
                os.remove(join(Paths.PICKLE_DIR, file))
        self.__update_size__(0)
    
#ITEM FETCHING
    #TODO: IMPLEMENT
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
    
    def __get_saved_index__(self, index) -> pd.DataFrame:
        for df in self.__loaded_chunks:
            if not df.empty and int(df[Keys.TIME_INTERVAL].iloc[-1]) >= index and int(df[Keys.TIME_INTERVAL].iloc[0]) <= index:
                return df
        return pd.DataFrame()

    def __add_chunk_to_saved__(self, chunk, index):
        if  not self.__get_saved_index__(index).empty:
            print("Element already loaded!")
            return
    
        #Add chunk to queue/ replace lastly added element if max_chunks size is reached
        print("Element not in loaded, adding...")

        self.__loaded_chunks[self.__loaded_chunks_next] = chunk
        #Update next in "queue"
        self.__loaded_chunks_next += 1
        if(self.__loaded_chunks_next >= self.max_chunks):
            self.__loaded_chunks_next = 0

    def __fetch_chunk__(self, index):
        '''Returns a dataframe with headers, containing data from picklefile that contains the given index'''
        if index < 0:
            return
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
        self.__add_chunk_to_saved__(loaded, index)
        return loaded





pklst = PickleDataset(max_saved_chunks=2)
#pklst.__del_db__()

pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)

pklst.__fetch_chunk__(1383865800000)
pklst.__fetch_chunk__(1383865800000)
pklst.__fetch_chunk__(1383865800000)
pklst.__fetch_chunk__(1383865800000)

pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)
pklst.__fetch_chunk__(1383260400000)

pklst.__fetch_chunk__(1383865800000)
pklst.__fetch_chunk__(1383865800000)
pklst.__fetch_chunk__(1383865800000)
pklst.__fetch_chunk__(1383865800000)