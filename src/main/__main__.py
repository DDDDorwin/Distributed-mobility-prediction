import sys
from pickleset import *
from models.example_model.modeled import example_model as models

from data.preprocessing_scripts import *

def main():
    # print(models(sys.argv[1]))
    #replace_nan_with_zeroes()
    from datetime import datetime
    from random import seed
    from random import randint

    pds = Pickle2dCNN(train_size=5,test_size=2,max_saved_chunks=8)
    seed(1)
    now = datetime.now()
    for i in range(200):
        rand_index = randint(0, pds.__len__())
        pds.__sliding_window__(rand_index)
    then = datetime.now()
    print("Time taken = ", then-now)

main()
