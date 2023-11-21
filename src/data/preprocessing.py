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

def normalize(input_dir, output_dir, destroy_old, normalize_cols: Iterable[str] = [
        'sms_in',
        'sms_out',
        'call_in',
        'call_out',
        'internet'
        ]):
    

    if destroy_old:
        del_f(output_dir)

    input_files = [
        join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")
    ]

    max_vals = [0] * len(normalize_cols)

    for input_file in sorted(input_files):
        df = pd.read_csv(input_file, header=None, sep="\t", dtype=TableData.DTYPES)
        for col in range(len(normalize_cols)):
            max_vals[col] = max(max_vals[col], df[normalize_cols[col]].max() )
        print("Successfully normalized %s" % (input_file))
    
    for i in max_vals:
        print(i)
    
    print("Finnished normalization")





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

    meta_file = join(input_dir, META.FILE_NAME)
    with open(meta_file, "r") as f:
        meta = json.load(f)

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

    meta_file = join(input_dir, META.FILE_NAME)
    with open(meta_file, "r") as f:
        meta = json.load(f)
    
    groupby_cols = [Keys.SQUARE_ID, Keys.TIME_INTERVAL]
    agg_cols = [Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT, Keys.INTERNET]
    agg_method = 'sum'

    for f in [f for f in os.listdir(input_dir)[:3] if f != META.FILE_NAME]:
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


def save_metafile(output_dir: str, df: pd.DataFrame):
    with open(join(output_dir, META.FILE_NAME), "w") as f:
        json.dump({META.DTYPES: df.dtypes.apply(lambda x: x.name).to_dict()}, f)
