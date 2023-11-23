import sys
from models.example_model.modeled import example_model as models

from data.preprocessing_scripts import *

def main():
    # print(models(sys.argv[1]))
    replace_nan_with_zeroes()

main()
