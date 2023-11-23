from utils.constants import *
from data.preprocessing import *

def group_cc():
    eliminate_country_code(Paths.RAW_DIR, Paths.GROUPED_CC_DIR)

def normalalize_cols():
    normalize(Paths.RAW_DIR, Paths.NORM_DIR, True)

def group_square_id():
    eliminate_square_id(Paths.RAW_DIR, Paths.GENERAL_DIR)

def replace_nan_with_zeroes():
    replace_null()
