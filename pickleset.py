import os 
import pandas as pd
import numpy as np
import queue
from data.data import Dataset
from typing import TypeVar
from constants import *
from os.path import join


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class PickleDataset(Dataset):
    is_loading = False

    def __init__(self, train_size, test_size, max_saved_chunks):
        self.__get_size__()
        if max_saved_chunks < 1:
            max_saved_chunks = 1
        self.max_chunks = max_saved_chunks
        self.__loaded_chunks = [pd.DataFrame] * max_saved_chunks
        self.__loaded_chunks_next = 0
        self.train_size = train_size
        self.test_size = test_size

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
        return self.size

    def __len__(self):
        return self.size

#CREATION/DELETION OF PICKLES
    def __make_pickles__(self, destroy_old):
        '''Create pickle files for each raw data file, name = startIndex_endIndex.pkl'''
        if destroy_old:
            #Delete existing pickle files
            self.__del_db__()
        #For all data files ending with .txt, .tsv or .csv, reformat them into pickles, adding corresponding headers in the process
        input_files = [join(Paths.RAW_DIR, f) for f in os.listdir(Paths.RAW_DIR) if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")]
        size = 0
        for input_file in sorted(input_files):
            #Read csv
            df = pd.read_csv(input_file, header=None, sep='\t', dtype=TableData.DTYPES)
            length = len(df)
            #Add corresponding indexes to rows
            df[Keys.INDEX] = [i for i in range(size, size + length)]
            df = df.set_index(Keys.INDEX)
            #Fetches keys for dtypes, to use as names for headers
            df.columns = TableData.DTYPES.keys()
            #Make pickles with name: startIndex_endIndex.pkl 
            df.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (size,size + length - 1)))
            #Increment size
            size += length
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
    def __getitem__(self, index) -> pd.DataFrame:
        chunk = self.__fetch_chunk__(index)
        return chunk.loc[[index]]
    
    def __sliding_window__(self, index):
         # Lists to store DataFrames
        train_window = []
        for train_offset in range(self.train_size):
            train_window.append(self.__getitem__(index + train_offset))
        test_window = []
        for test_offset in range(self.test_size):
            test_window.append(self.__getitem__(index + self.train_size + test_offset))

        return {'train': pd.concat(train_window, ignore_index=True), 'test': pd.concat(test_window, ignore_index=True)}
    
    def __get_saved_index__(self, index) -> pd.DataFrame:
        for df in self.__loaded_chunks:
            if not df.empty and index in df.index:
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
            print("Index found in loaded, no fetch needed.")
            return chunk
        loaded = pd.DataFrame()
        print("Index was not found in loaded, fetching...")
        #Get missing pickle file in directory
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        for p in pickles:
            indexes = p.replace(".pkl",'').split("_", 1)
            #Check if file contains data in range
            if(int(indexes[1]) >= index and int(indexes[0]) <= index):
                #Append to return dataframe
                loaded = pd.read_pickle(join(Paths.PICKLE_DIR, p))
                print("Index contained in pickle: %s" % (p))

        #If nothing is being prefetched, save chunk
        self.__add_chunk_to_saved__(loaded, index)
        return loaded



'''

pklst = PickleDataset(train_size=5,test_size=5,max_saved_chunks=2)
#pklst.__make_pickles__(True)
#pklst.__del_db__()
print(pklst.__len__())

pklst.__getitem__(0)
pklst.__getitem__(0)
pklst.__getitem__(0)
pklst.__getitem__(0)
pklst.__getitem__(0)

pklst.__getitem__(37622898)
pklst.__getitem__(37622898)
pklst.__getitem__(37622900)
pklst.__getitem__(43541753)
pklst.__getitem__(43541754)


pklst.__getitem__(37622898)

print(pklst.__getitem__(0))

print(pklst.__sliding_window__(0))
'''


