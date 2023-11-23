from utils.constants import *
from data.preprocessing import *

def group_cc():
    eliminate_country_code()

def normalalize_cols():
    normalize()

def group_square_id():
    eliminate_square_id()

def replace_nan_with_zeroes():
    replace_null()

def group_and_normalize():
    replace_null()
    eliminate_country_code(input_dir=Paths.FILLED_WITH_ZEROES_DIR)
    eliminate_square_id(input_dir=Paths.Paths.GROUPED_CC_DIR)
    normalize(input_dir=Paths.GENERAL_DIR, output_dir=Paths.PREPROCESSED_DIR)
