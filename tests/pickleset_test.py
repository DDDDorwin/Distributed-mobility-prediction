import unittest
import pandas as pd
import numpy as np
import os

import pytest

from utils.constants import Paths
from src.pickleset.pickleset import PickleDataset
from random import seed
from random import randint
from os.path import join

"""
This file contains tests to ensure function of the pickleset dataset.
The tests are testing all functions of the dataset, including the following:
    TestPickleSetFuncs
        - size correctness / in bounds
        - I/O of size file "SIZE_DATA.txt"
        - size changes
    TestPickleBuildFuncs
        - deletion and creation
        - makes sure dataset is initialized propperly
    TestFetching
        - test getitem consistency
        - test getitem returns a dataframe of size 1
"""

nChunks = 8


@pytest.mark.skip(reason="no way of currently testing this")
class TestPickleSetFuncs(unittest.TestCase):
    """Test cases for size functions of dataset."""

    def __get_actuall_size(self):
        """Helper function for getting the actual number of rows in the database."""
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        length = 0
        for p in pickles:
            length += len(pd.read_pickle(join(Paths.PICKLE_DIR, p)))
        return length

    def test_size(self):
        """Testing of the ___len__() function, and the ___get_size___() function."""
        pds = PickleDataset(train_size=4, test_size=2, max_saved_chunks=nChunks)
        size = pds.__len__()
        self.assertEqual(pds._PickleDataset__get_size(), size)
        self.assertEqual(pds._PickleDataset__get_size(), self.__get_actuall_size())

    def test_change_size(self):
        """Test to increase the size and then decrease it."""
        pds = PickleDataset(train_size=4, test_size=2, max_saved_chunks=nChunks)
        offset = 1932
        size = pds.__len__()
        pds._PickleDataset__update_size(size + offset)
        self.assertEqual(pds._PickleDataset__get_size(), size + offset)
        pds._PickleDataset__update_size(size)
        self.assertEqual(pds._PickleDataset__get_size(), self.__get_actuall_size())

    def test_assert_size_in_bounds(self):
        """Test that the size is not negative, or bigger than the actual number of rows in the pickle directory."""
        pds = PickleDataset(train_size=4, test_size=2, max_saved_chunks=nChunks)
        self.assertGreaterEqual(pds.__len__(), 0)
        self.assertLessEqual(pds.__len__(), self.__get_actuall_size())


@pytest.mark.skip(reason="no way of currently testing this")
class TestPickleBuildFuncs(unittest.TestCase):
    def __get_actuall_size(self):
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        length = 0
        for p in pickles:
            length += len(pd.read_pickle(join(Paths.PICKLE_DIR, p)))
        return length

    def test_deletion_creation(self):
        # Delete phase
        n_files = len([f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")])
        pds = PickleDataset(train_size=4, test_size=2, max_saved_chunks=nChunks)
        pds._PickleDataset__del_db()
        no_files = len([f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")])
        self.assertEqual(0, no_files)
        self.assertEqual(0, pds.__len__())

        # Create phase
        pds._PickleDataset__make_pickles(False)
        self.assertEqual(n_files, len([f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]))
        self.assertEqual(pds.__len__(), self.__get_actuall_size())


@pytest.mark.skip(reason="no way of currently testing this")
class TestFetching(unittest.TestCase):
    def test_get_consistent(self):
        """Test that __getitem__() returns consistent values"""
        pds = PickleDataset(train_size=4, test_size=2, max_saved_chunks=nChunks)
        seed(1)
        for i in range(100):
            rand_index = randint(0, pds.__len__())
            self.assertTrue(pds.__getitem__(rand_index).equals(pds.__getitem__(rand_index)))

    def test_get_check_length(self):
        """Tests that the items returned from __getitem__() are of length one (one row) and contain 8 columns"""
        pds = PickleDataset(train_size=4, test_size=2, max_saved_chunks=nChunks)
        seed(2)
        for i in range(100):
            rand_index = randint(0, pds.__len__())
            self.assertEqual(len(pds.__getitem__(rand_index)), 1)
            self.assertEqual(len(pds.__getitem__(rand_index).columns), 8)

    def test_sliding_window(self):
        """Tests that the sliding window algorithm returns train/test sets of correct size and type."""
        nTr = 4
        nTe = 2
        pds = PickleDataset(train_size=nTr, test_size=nTe, max_saved_chunks=nChunks)
        seed(3)
        for i in range(100):
            rand_index = randint(0, pds.__len__())
            arr = pds.sliding_window(rand_index)
            self.assertEqual(len(arr), 2)
            self.assertIsInstance(arr[0], np.ndarray)
            self.assertIsInstance(arr[1], np.ndarray)
            self.assertEqual(len(arr[0]), nTr)
            self.assertEqual(len(arr[1]), nTe)


# TODO: MAKE TESTS FOR EDGE CASES!!! ESP FOR SLIDING WINDOW
# ALSO DO TESTS FOR CHUNKS


if __name__ == "__main__":
    unittest.main()
