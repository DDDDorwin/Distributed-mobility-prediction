"""
This module is to define constants, mainly string keys. Code should be clear
from all magic string keys and register them instead here. The classes are
there to distinguish between topics and hold the constants as class variables.
They can be used in one another but it is not as powerful as e.g. properties.

Class names should be written in CamelCase
Constant names should be written in ALL_CAPS.

Ideally never use hard-coded strings or numbers in your code, put them here,
and import this module in your script instead:

from constants import *

or

from utils import Keys

etc. Then use them in your code as e.g.

Keys.SQUARE_ID
"""

from os.path import join as pjoin


class Paths(object):
    """File and directory paths on uppmax"""

    # directories
    DATA_DIR = pjoin("proj", "uppmax2023-2-33", "nobackup", "data")
    # DATA_DIR = "/home/final/projects/data/"
    # DATA_DIR = "/Users/savvas/Desktop/PCS/Project_CS_UserVsSpecific/data"
    RAW_DIR = pjoin(DATA_DIR, "raw")
    
    NORM_DIR = pjoin(DATA_DIR, "norm_raw")
    PICKLE_DIR = pjoin(DATA_DIR, "pickles")
    FILLED_WITH_ZEROES_DIR = pjoin(DATA_DIR, "filled_with_zeroes")
    GENERAL_DIR = pjoin(DATA_DIR, "general")
    FILTERED_DIR = pjoin(DATA_DIR, "filtered")
    DATETIME_DIR = pjoin(DATA_DIR, "datetime")
    PREPROCESSED_DIR = pjoin(DATA_DIR, "preprocessed")
    GROUPED_CC_DIR = pjoin(PREPROCESSED_DIR, "grouped_country_codes")
    MERGED_TO_SIZE_DIR = pjoin(PREPROCESSED_DIR, "merged_to_size")

    # files
    MERGED_RAW_F = pjoin(PREPROCESSED_DIR, "raw_merged.tsv")
    GROUPED_CC_F = pjoin(PREPROCESSED_DIR, "grouped_cc.pkl")
    GROUPED_TIME_F = pjoin(PREPROCESSED_DIR, "grouped_time.pkl")
    MILANO_GRID_GEOJSON = "assets/milano-grid.geojson"
    TWO_WEEKS_CELL_ACTIVITY_DATA_PATH = pjoin(DATA_DIR, "two_weeks_cell.csv")

    # test
    TEST_DIR = pjoin("./test")


class Keys(object):
    """Column names for dataframes"""

    # raw data set
    COORDINATE = "coordinate"
    CELL_TRAFFIC = "cell_traffic"
    DATE = "date"
    INDEX = "index"
    SQUARE_ID = "square_id"
    TIME_INTERVAL = "time_interval"
    COUNTRY_CODE = "country_code"
    SMS_IN = "sms_in"
    SMS_OUT = "sms_out"
    CALL_IN = "call_in"
    CALL_OUT = "call_out"
    INTERNET = "internet"


class META:
    "Keys for the data directory meta files"

    FILE_NAME = "meta.json"
    DTYPES = "dtypes"


class TableData(object):
    """Numpy/Pandas datatypes and other descriptors for columns in dataframes

    Values should be string names of pandas dtypes
    https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
    """

    # raw data set
    DTYPES = {
        Keys.SQUARE_ID: "int16",
        Keys.TIME_INTERVAL: "int64",
        Keys.COUNTRY_CODE: "int8",
        Keys.SMS_IN: "float64",
        Keys.SMS_OUT: "float64",
        Keys.CALL_IN: "float64",
        Keys.CALL_OUT: "float64",
        Keys.INTERNET: "float64",
    }
