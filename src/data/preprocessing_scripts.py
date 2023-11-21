from constants import *
from .preprocessing import *

def group_cc():
    eliminate_country_code(Paths.RAW_DIR, Paths.GROUPED_CC_DIR)

def normalalize_cols():
    normalize(Paths.RAW_DIR, Paths.NORM_DIR, True)
