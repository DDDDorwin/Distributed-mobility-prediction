import sys
from pickleset.pickleset import *
from models.example_model.modeled import example_model as models

from data.preprocessing_scripts import *

def main():
    # print(models(sys.argv[1]))
    #replace_nan_with_zeroes()
    from datetime import datetime
    from random import seed
    from random import randint

    pds = Pickle2dCNN(train_size=9,test_size=2,max_saved_chunks=2)
    seed(1)
    now = datetime.now()
    rand_index = randint(0, pds.__len__())
    print(pds.__getitem__(rand_index))
    then = datetime.now()
    print("Time taken = ", then-now)
    