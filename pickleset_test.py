import unittest
from constants import *
from pickleset import *
from random import seed
from random import randint

'''
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
'''

class TestPickleSetFuncs(unittest.TestCase):
    '''Test cases for size functions of dataset.'''

    def __get_actuall_size(self):
        '''Helper function for getting the actual number of rows in the database.'''
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        length = 0
        for p in pickles:
            length += len(pd.read_pickle(join(Paths.PICKLE_DIR, p)))
        return length

    def test_size(self):
        '''Testing of the ___len__() function, and the ___get_size___() function.'''
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=2)
        size = pds.__len__()
        self.assertEqual(pds.__get_size__(), size)
        self.assertEqual(pds.__get_size__(), self.__get_actuall_size())

    def test_change_size(self):
        '''Test to increase the size and then decrease it.'''
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=2)
        offset = 1932
        size = pds.__len__()
        pds.__update_size__(size+offset)
        self.assertEqual(pds.__get_size__(), size+offset)
        pds.__update_size__(size)
        self.assertEqual(pds.__get_size__(), self.__get_actuall_size())

    def test_assert_size_in_bounds(self):
        '''Test that the size is not negative, or bigger than the actual number of rows in the pickle directory.'''
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=2)
        self.assertGreaterEqual(pds.__len__(), 0)
        self.assertLessEqual(pds.__len__(), self.__get_actuall_size())


class TestPickleBuildFuncs(unittest.TestCase):

    def __get_actuall_size(self):
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        length = 0
        for p in pickles:
            length += len(pd.read_pickle(join(Paths.PICKLE_DIR, p)))
        return length

    def test_deletion_creation(self):
        #Delete phase
        n_files = len([f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")])
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=2)
        pds.__del_db__()
        no_files = len([f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")])
        self.assertEqual(0, no_files)
        self.assertEqual(0, pds.__len__())

        #Create phase
        pds.__make_pickles__(False)
        self.assertEqual(n_files, len([f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]))
        self.assertEqual(pds.__len__(), self.__get_actuall_size())

class TestFetching(unittest.TestCase):

    def test_get_consistent(self):
        '''Test that __getitem__() returns consistent values'''
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=10)
        seed(1)
        for i in range(100):
            rand_index = randint(0, pds.__len__())
            self.assertTrue(pds.__getitem__(rand_index).equals(pds.__getitem__(rand_index)))

    def test_get_check_length(self):
        '''Tests that the items returned from __getitem__() are of length one (one row)'''
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=10)
        seed(2)
        for i in range(100):
            rand_index = randint(0, pds.__len__())
            self.assertEqual(len(pds.__getitem__(rand_index)), 1)
    

#TODO: MAKE TESTS FOR EDGE CASES!!! ESP FOR SLIDING WINDOW
# ALSO DO TESTS FOR CHUNKS


if __name__ == '__main__':
    unittest.main()