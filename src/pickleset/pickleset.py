import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.constants import Paths, Keys
from os.path import join

class PickleDataset(Dataset):

    def __init__(self, train_size, test_size, max_saved_chunks, raw_directory:str = Paths.RAW_DIR, pickle_dir:str = Paths.PICKLE_DIR):
        self.raw_dir = raw_directory
        self.p_dir = pickle_dir
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
        f = open(join(self.p_dir, "SIZE_DATA.txt"), "w+")
        f.write("%d" % size)
        f.close()
        self.size = size

    def __get_size(self):
        """Fetch the current size from SIZE_DATA.txt and set to self.size"""
        # If SIZE_DATA.txt exists, get size and set to self.size
        if not os.stat(join(self.p_dir, "SIZE_DATA.txt")).st_size == 0:
            f = open(join(self.p_dir, "SIZE_DATA.txt"), "r")
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
            join(self.raw_dir, f)
            for f in os.listdir(self.raw_dir)
            if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")
        ]
        size = 0
        for input_file in sorted(input_files):
            # Read "input file" as csv
            df = pd.read_csv(input_file, header=None, sep="\t")
            length = len(df)
            # Add corresponding indexes to rows
            df[Keys.INDEX] = [i for i in range(size, size + length)]
            df = df.set_index(Keys.INDEX)
            # Make pickles with name: startIndex_endIndex.pkl
            df.to_pickle(join(self.p_dir, "%s_%s.pkl" % (size, size + length - 1)))
            # Increment size
            size += length
            print("Successfully pickled %s" % (input_file))
        self.__update_size(size)

    def __del_db(self):
        """Deletes all pickles from pickles directory"""
        for file in os.listdir(self.p_dir):
            if os.fsdecode(file).endswith(".pkl"):
                os.remove(join(self.p_dir, file))
        self.__update_size(0)

    # ITEM FETCHING
    def __getitem__(self, index):
        """Returns a dataframe with one row containing the found item."""
        chunk = self.__fetch_chunk(index)
        return chunk.loc[[index]]

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
        pickles = [f for f in os.listdir(self.p_dir) if f.endswith(".pkl")]
        for p in pickles:
            indexes = p.replace(".pkl", "").split("_", 1)
            # Check if file contains data in range
            if int(indexes[1]) >= index and int(indexes[0]) <= index:
                # Append to return dataframe
                loaded = pd.read_pickle(join(self.p_dir, p))
                print("Index contained in pickle: %s" % (p))

        self.__add_chunk_to_saved(loaded, index)
        return loaded


class Pickle2dCNN(PickleDataset):
    def __init__(self, train_size, test_size, max_saved_chunks, raw_directory:str = Paths.RAW_DIR, pickle_dir:str = Paths.PICKLE_DIR):    
        super().__init__(train_size, test_size, max_saved_chunks, raw_directory, pickle_dir)

    def __getitem__(self, index):
        """Returns a tensor with the shape [[[a1,a2,a3...],[a4,a5...]],[[b1,b2,b3...],[b4,b5...]]...] where [a1, a2, a3] and [b1, b2, b3]
        are feature columns from the dataset used for training, and [a4, a5] and [b4, b5] are feature columns used for testing.
        
        a1, b1 etc. are values of the rows 'index', and a2, b2 etc. are the values of the rows 'index + 1'.
        """
        all = []
        for offset in range(self.train_size + self.test_size):
            all.append(super().__getitem__(index + offset).values)

        tensor = torch.tensor(np.array(all))
        t1 = torch.rot90(tensor[0:self.train_size, :], k=1, dims=(1, 0)).mT
        t2 = torch.rot90(tensor[self.train_size:self.train_size + self.test_size, :], k=1, dims=(1, 0)).mT

        return [t1, t2]
    
class PickleARIMA(PickleDataset):
    def __init__(self, train_size, test_size, max_saved_chunks, raw_directory:str = Paths.RAW_DIR, pickle_dir:str = Paths.PICKLE_DIR):    
        super().__init__(train_size, test_size, max_saved_chunks, raw_directory, pickle_dir)

    def __getitem__(self, index):
        """Returns a tensor with the shape [[[a1,a2,a3...],[a4,a5...]],[[b1,b2,b3...],[b4,b5...]]] where [a1, a2, a3] and [b1, b2, b3]
            are feature columns from the dataset used for training, and [a4, a5] and [b4, b5] are feature columns used for testing.
          
            a1, b1 etc. are values of the rows 'index', and a2, b2 etc. are the values of the rows 'index + 1'.

            Specific columns descriptions:\n
            \tValues starting with a = time\n
            \tValues starting with b = internet
          """
        None
