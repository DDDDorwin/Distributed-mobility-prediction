import sys
from models.example_model.modeled import example_model as models


def main():
    print(models(sys.argv[1]))


main()
