import unittest
import sys
sys.path.append(".")
from constants import *
from pickleset import *

class TestPickleSetFuncs(unittest.TestCase):

    def __get_actuall_size(self):
        pickles = [f for f in os.listdir(Paths.PICKLE_DIR) if f.endswith(".pkl")]
        length = 0
        for p in pickles:
            length += len(pd.read_pickle(join(Paths.PICKLE_DIR, p)))
        return length

    def test_size(self):
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=2)
        size = pds.__len__()
        self.assertEqual(pds.__get_size__(), size)
        self.assertEqual(pds.__get_size__(), self.__get_actuall_size())

    def test_change_size(self):
        '''Test to increase the size and then decrease it'''
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=2)
        offset = 1932
        size = pds.__len__()
        pds.__update_size__(size+offset)
        self.assertEqual(pds.__get_size__(), size+offset)
        pds.__update_size__(size)
        self.assertEqual(pds.__get_size__(), self.__get_actuall_size())

    def assert_size_in_bounds(self):
        pds = PickleDataset(train_size=4,test_size=2,max_saved_chunks=2)
        self.assertGreaterEqual(pds.__len__, 0)
        self.assertLessEqual(pds.__len__, self.__get_actuall_size())

    


    


class TestPickleBuildFuncs(unittest.TestCase):
    None

if __name__ == '__main__':
    unittest.main()