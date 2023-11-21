from constants import *
from .preprocessing import *

def group_cc():
    eliminate_country_code(Paths.RAW_DIR, Paths.GROUPED_CC_DIR)
