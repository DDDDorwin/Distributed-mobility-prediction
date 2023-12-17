from gvslearning.data.preprocessing import *
from gvslearning.utils.constants import *


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
    eliminate_square_id(input_dir=Paths.GROUPED_CC_DIR)
    normalize(input_dir=Paths.GENERAL_DIR, output_dir=Paths.PREPROCESSED_DIR)


def filter_square_ids():
    # example: 100 values between 1 .. 10000
    selected_values = [int(1 + i * 101) for i in range(100)]
    filter_rows(columns_values={Keys.SQUARE_ID: selected_values})


def timestamp_to_datetime():
    convert_timestamp()
