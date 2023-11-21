import os
import torch
import json
import pandas as pd
from typing import TypeVar, Iterable, Dict
from constants import *
from os.path import join


def del_f(destroy_dir):
    for file in os.listdir(destroy_dir):
        if os.fsdecode(file).endswith(".txt"):
            os.remove(join(destroy_dir, file))


def normalize(input_dir: str, output_dir: str, destroy_old: bool = True, 
              normalize_cols: Iterable[str] = [Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT, Keys.INTERNET],
              norm_upper:float = 1, norm_lower:float = 0
              ) -> None:
    '''
    Group all rows by square_id, and/or time_interval then
    remove country codes when aggregating CDRs.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if destroy_old:
        del_f(output_dir)

    meta = load_metafile(input_dir)

    max_val = 0

    for f in [f for f in os.listdir(input_dir) if f != META.FILE_NAME]:
        df = load_textfile(join(input_dir, f), dtypes=meta[META.DTYPES])
        for col in range(len(normalize_cols)):
            max_val = max(max_val, df[normalize_cols[col]].max())
    
    def norm(x):
        return norm_lower + (x + norm_upper - norm_lower)/max_val
    
    for f in [f for f in os.listdir(input_dir) if f != META.FILE_NAME]:
        df = load_textfile(join(input_dir, f), dtypes=meta[META.DTYPES])
        df[normalize_cols] = df[normalize_cols].apply(norm)
        save_textfile(join(output_dir, f), df)
    save_metafile(output_dir, df)


def groupby_agg(df: pd.DataFrame, groupby_cols: Iterable[str], agg_cols: Iterable[str], agg_method: str) -> pd.DataFrame:
    '''Pipe groupby and aggregate after filtering for existing columns.'''

    # check if cols to group and aggregate actually exist
    groupby_cols = [k for k in groupby_cols if k in df]
    agg_cols = [k for k in agg_cols if k in df]

    # aggregate rows
    # this will automatically fill some NaNs with zeroes
    return df.groupby(groupby_cols) \
             .agg({k: agg_method for k in agg_cols}) \
             .reset_index()

def eliminate_country_code(input_dir: str, output_dir: str, destroy_old: bool = True) -> None:
    '''
    Group all rows by square_id, and/or time_interval then
    remove country codes when aggregating CDRs.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if destroy_old:
        del_f(output_dir)

    meta = load_metafile(input_dir)
    
    groupby_cols = [Keys.SQUARE_ID, Keys.TIME_INTERVAL]
    agg_cols = [Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT, Keys.INTERNET]
    agg_method = 'sum'

    for f in [f for f in os.listdir(input_dir) if f != META.FILE_NAME]:
        df = load_textfile(join(input_dir, f), dtypes=meta[META.DTYPES])
        df = groupby_agg(df, groupby_cols, agg_cols, agg_method)
        save_textfile(join(output_dir, f), df)

    save_metafile(output_dir, df)


def eliminate_square_id(input_dir: str, output_dir: str, destroy_old: bool = True) -> None:
    '''
    Group all rows by time_interval and/or country_code then
    remove square ids when aggregating CDRs.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if destroy_old:
        del_f(output_dir)

    meta = load_metafile(input_dir)
    
    groupby_cols = [Keys.TIME_INTERVAL, Keys.COUNTRY_CODE]
    agg_cols = [Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT, Keys.INTERNET]
    agg_method = 'sum'

    for f in [f for f in os.listdir(input_dir) if f != META.FILE_NAME]:
        df = load_textfile(join(input_dir, f), dtypes=meta[META.DTYPES])
        df = groupby_agg(df, groupby_cols, agg_cols, agg_method)
        save_textfile(join(output_dir, f), df)

    save_metafile(output_dir, df)


def load_textfile(input_file: str, dtypes: Dict[str, str]):
    '''
    Load tsv from an input file and put it into a dataframe.
    Takes a long time and a lot of memory.
 
    input_file: str -- name of input file
    parse_dates: bool -- convert unix timestamp to pandas datetime
    '''
    df = pd.read_csv(
        input_file,
        sep='\t', header=None, dtype=dtypes
    )
    df.columns = list(dtypes.keys())

    return df


def save_textfile(output_file: str, df: pd.DataFrame):
    df.to_csv(output_file, sep='\t', header = False, index = False)
    print(f"saved {output_file}")


def load_metafile(dir: str):
    """
    Load the metafile of a data set directory.
    Provides a fallback with default values if the file
    does not exist.
    """
    
    meta_file = join(dir, META.FILE_NAME)
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)
    else:
        meta = {META.DTYPES: TableData.DTYPES}
    return meta

def save_metafile(output_dir: str, df: pd.DataFrame):
    """Generate a metafile for a data set directory."""
    with open(join(output_dir, META.FILE_NAME), "w") as f:
        json.dump({META.DTYPES: df.dtypes.apply(lambda x: x.name).to_dict()}, f)
