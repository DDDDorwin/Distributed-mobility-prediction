import sys
from pickleset.pickleset import *
from models.example_model.modeled import example_model as models
from datetime import datetime
from random import seed
from random import randint

from data.preprocessing_scripts import *

def main():
    # print(models(sys.argv[1]))
    
    ## Preprocessing 
    # replace_nan_with_zeroes()
    # filter_square_ids()
    
    ## Pickle Set
    # pds = Pickle2dCNN(train_size=9,test_size=2,max_saved_chunks=2)
    # seed(1)
    # now = datetime.now()
    # rand_index = randint(0, pds.__len__())
    # print(pds.__getitem__(rand_index))
    # then = datetime.now()
    # print("Time taken = ", then-now)

    pass

if __name__ == "__main__":
    main()
