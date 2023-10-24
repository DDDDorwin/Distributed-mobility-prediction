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

from constant import Keys

etc. Then use them in your code as e.g.

Keys.SQUARE_ID
"""


from os.path import join as pjoin


class Paths(object):
    """File and directory paths on uppmax"""

    # directories
    DATA_DIR = pjoin("proj", "uppmax2023-2-33", "nobackup", "data")
    RAW_DIR = pjoin(DATA_DIR, "raw")
    PICKLE_DIR = pjoin(DATA_DIR, "pickles")
    PREPROCESSED_DIR = pjoin(DATA_DIR, "preprocessed")
    GROUPED_CC_DIR = pjoin(PREPROCESSED_DIR, "grouped_country_codes")
    MERGED_TO_SIZE_DIR = pjoin(PREPROCESSED_DIR, "merged_to_size")

    # files
    MERGED_RAW_F = pjoin(PREPROCESSED_DIR, 'raw_merged.tsv')
    GROUPED_CC_F = pjoin(PREPROCESSED_DIR, 'grouped_cc.pkl')
    GROUPED_TIME_F = pjoin(PREPROCESSED_DIR, 'grouped_time.pkl')

    # test
    TEST_DIR = pjoin('./test')


class Keys(object):
    """Column names for dataframes"""

    # raw data set
    SQUARE_ID = "square_id"
    TIME_INTERVAL = "time_interval"
    COUNTRY_CODE = "country_code"
    SMS_IN = "sms_in"
    SMS_OUT = "sms_out"
    CALL_IN = "call_in"
    CALL_OUT = "call_out"
    INTERNET = "internet"



class TableData(object):
    """Numpy/Pandas datatypes and other descriptors for columns in dataframes

    Values should be string names of pandas dtypes
    https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
    """

    # index of each column in the raw data set
    INDICES = {
        Keys.SQUARE_ID: 0,
        Keys.TIME_INTERVAL: 1,
        Keys.COUNTRY_CODE: 2,
        Keys.SMS_IN: 3,
        Keys.SMS_OUT: 4,
        Keys.CALL_IN: 5,
        Keys.CALL_OUT: 6,
        Keys.INTERNET: 7,
    }

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
