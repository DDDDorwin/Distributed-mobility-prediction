import os
import torch
import math
import numpy as np
import pandas as pd
from data.data import Dataset
from typing import TypeVar
from constants import Paths, Keys, TableData
from os.path import join


T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class PickleDataset(Dataset):

    is_loading = False
    nFeatures = 0

    def __init__(self, train_size, test_size, max_saved_chunks):
        self.__get_size()
        if max_saved_chunks < 1:
            max_saved_chunks = 1
        self.max_chunks = max_saved_chunks
        self.__loaded_chunks = [pd.DataFrame] * max_saved_chunks
        self.__loaded_chunks_next = 0
        self.train_size = train_size
        self.test_size = test_size
        print("%s pickled rows of data exist!" % self.size)

    # SIZE RELATED FUNCTIONS
    def __update_size(self, size):
        """Update size in SIZE_DATA.txt and update self.size"""
        f = open(join(Paths.PICKLE_DIR, "SIZE_DATA.txt"), "w+")
        f.write("%d" % size)
        f.close()
        self.size = size

    def __get_size(self):
        """Fetch the current size from SIZE_DATA.txt and set to self.size"""
        # If SIZE_DATA.txt exists, get size and set to self.size
        if not os.stat(join(Paths.PICKLE_DIR, "SIZE_DATA.txt")).st_size == 0:
            f = open(join(Paths.PICKLE_DIR, "SIZE_DATA.txt"), "r")
            self.size = int(f.readline())
            f.close()
        # If there is no file SIZE_DATA.txt, we create one and set size to 0
        else:
            self.__update_size(0)
        return self.size

    def __len__(self):
        """Returns the length of the pickleset"""
        return self.size

    # CREATION/DELETION OF PICKLES
    def __make_pickles(self, destroy_old):
        """Create pickle files for each raw data file, name = startIndex_endIndex.pkl"""
        if destroy_old:
            # Delete existing pickle files
            self.__del_db()
        # For all data files ending with .txt, .tsv or .csv
        # reformat them into pickles, adding corresponding headers in the process
        input_files = [
            join(Paths.RAW_DIR, f)
            for f in os.listdir(Paths.RAW_DIR)
            if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")
        ]
        size = 0
        for input_file in sorted(input_files):
            # Read "input file" as csv
            df = pd.read_csv(input_file, header=None, sep="\t", dtype=TableData.DTYPES)
            length = len(df)
            # Add corresponding indexes to rows
            df[Keys.INDEX] = [i for i in range(size, size + length)]
            df = df.set_index(Keys.INDEX)
            # Fetches keys for dtypes, to use as names for headers
            df.columns = TableData.DTYPES.keys()
            # Make pickles with name: startIndex_endIndex.pkl
            df.to_pickle(join(Paths.PICKLE_DIR, "%s_%s.pkl" % (size, size + length - 1)))
            # Increment size
            size += length
            print("Successfully pickled %s" % (input_file))
        self.__update_size(size)

    def __del_db(self):
        """Deletes all pickles from pickles directory"""
        for file in os.listdir(Paths.PICKLE_DIR):
            if os.fsdecode(file).endswith(".pkl"):
                os.remove(join(Paths.PICKLE_DIR, file))
        self.__update_size(0)

    # ITEM FETCHING
    # TODO: IMPLEMENT
    def __getitem__(self, index) -> pd.DataFrame:
        """Returns a tensor with one row containing the found item."""
        chunk = self.__fetch_chunk(index)
        return torch.tensor(chunk.loc[[index]].values)

    def sliding_window(self, index):
        """Returns an array containing a train [0] and a test [1] set as numpy arrays, created from the index given."""
        # Lists to store DataFrames
        train_window = []
        for train_offset in range(self.train_size):
            if(index + train_offset < self.size):
                train_window.append(self.__getitem__(index + train_offset))
        test_window = []
        for test_offset in range(self.test_size):
            if(index + self.train_size + test_offset < self.size):
                test_window.append(self.__getitem__(index + self.train_size + test_offset))

        if(len(test_window) > 0):
            return [
                pd.concat(train_window, ignore_index=False).to_numpy(),
                pd.concat(test_window, ignore_index=False).to_numpy(),
            ]
        return [pd.concat(train_window, ignore_index=False).to_numpy(), pd.DataFrame.empty]
    
    def sliding_window_by_feature_2dCNN(self, index):
        """Returns a tensor with the shape [[[a1,a2,a3...],[a4,a5...]],[[b1,b2,b3...],[b4,b5...]]...] where [a1, a2, a3] and [b1, b2, b3]
          are feature columns from the dataset used for training, and [a4, a5] and [b4, b5] are feature columns used for testing.
          
          a1, b1 etc. are values of the rows 'index', and a2, b2 etc. are the values of the rows 'index + 1'.

          Columns included, in the order they are included:
          \ttime\n
          \tsms_in\n
          \tsms_out\n
          \tcall_in\n
          \tcall_out\n
          \tinternet\n
          """
        all = []
        for offset in range(self.train_size + self.test_size):
            all.append(self.__getitem__(index + offset).drop(columns=[Keys.SQUARE_ID, Keys.COUNTRY_CODE], axis=0).values)

        tensor = torch.tensor(np.array(all))
        tensor = tensor.permute(0, 2, 1)
        return torch.split(torch.rot90(tensor, k=1, dims=(1, 0)).T, 2)




        
    
    def sliding_window_by_feature_ARIMA(self, index):
        """Returns a tensor with the shape [[[a1,a2,a3...],[a4,a5...]],[[b1,b2,b3...],[b4,b5...]]] where [a1, a2, a3] and [b1, b2, b3]
            are feature columns from the dataset used for training, and [a4, a5] and [b4, b5] are feature columns used for testing.
          
            a1, b1 etc. are values of the rows 'index', and a2, b2 etc. are the values of the rows 'index + 1'.

            Specific columns descriptions:\n
            \tValues starting with a = time\n
            \tValues starting with b = internet
          """
        None

    def __get_saved_index(self, index) -> pd.DataFrame:
        """Get the dataframe that contains the provided index. Empty DF if index does not exist."""
        for df in self.__loaded_chunks:
            if not df.empty and index in df.index:
                return df
        return pd.DataFrame()

    def __add_chunk_to_saved(self, chunk, index):
        """Adds the given chunk to the saved chunks FIFO queue."""
        # Check if element exists already.
        if not self.__get_saved_index(index).empty:
            print("Element already loaded!")
            return

        print("Element not in loaded, adding...")
        self.__loaded_chunks[self.__loaded_chunks_next] = chunk
        # Update next in "queue"
        self.__loaded_chunks_next += 1
        if self.__loaded_chunks_next >= self.max_chunks:
            self.__loaded_chunks_next = 0

    def __fetch_chunk(self, index):
        """Returns a dataframe with headers, containing data from picklefile that contains the given index."""
        if index < 0:
            return
        chunk = self.__get_saved_index(index)
        # Check if dataframe is already loaded for requested timeframe
        if len(chunk) > 0:
            print("Index found in loaded, no fetch needed.")
            return chunk
        loaded = pd.DataFrame()
        print("Index was not found in loaded, fetching...")
        # Get missing pickle file in directory
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        for p in pickles:
            indexes = p.replace(".pkl", "").split("_", 1)
            # Check if file contains data in range
            if int(indexes[1]) >= index and int(indexes[0]) <= index:
                # Append to return dataframe
                loaded = pd.read_pickle(join(Paths.PICKLE_DIR, p))
                print("Index contained in pickle: %s" % (p))

        self.__add_chunk_to_saved(loaded, index)
        return loaded


###BENCHMARK CODE:::::::::::::::###
from datetime import datetime
from random import seed
from random import randint

pds = PickleDataset(train_size=2,test_size=1,max_saved_chunks=16)
seed(1)
now = datetime.now()
for i in range(200):
    rand_index = randint(0, pds.__len__())
    print(pds.__getitem__(rand_index))
then = datetime.now()
print("Time taken = ", then-now)


"""
###BENCHMARK RESULTS:::::::::::::::###

GET ITEM

BENCHMARKING @200 iterations of __getitem__(), random indexes, 16 pickles, total 86,351,806 rows
nCHUNKS,    RAM USED (peak),    RUN TIME (s)
1           0.64GB              41.387061
2           1.10GB              38.822287
4           1.89GB              35.919033
8           3.63GB              24.747721
10          4.88GB              18.945788
12          5.91GB              14.359443
14          6.30GB              09.824801
16          8.07GB              04.012896


"""
