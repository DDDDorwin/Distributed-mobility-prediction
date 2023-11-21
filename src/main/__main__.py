import sys
from models.example_model.modeled import example_model


def main():
    print(example_model(sys.argv[1]))


main()
