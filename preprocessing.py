import os
import multiprocessing as mp
from os import listdir
from os.path import join, getsize
from typing import Iterable

import pandas as pd

from constants import Paths, Keys, TableData


def raw_tsv_to_sqlite(input_dir: str = Paths.RAW_DIR, output_dir: str = Paths.PREPROCESSED_DIR, output_file: str = 'raw.sqlite3'):
    '''Merge all tsv files into a single sqlite3 database file.'''
    
    import sqlite3

    # read tsv
    columns = [Keys.SQUARE_ID, Keys.TIME_INTERVAL, Keys.COUNTRY_CODE,
               Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT,
               Keys.INTERNET]
    
    # create data base file if it doesn't exist yet
    con = sqlite3.connect(join(output_dir, output_file))
    cur = con.cursor()
    cur.execute(f'''CREATE TABLE IF NOT EXISTS cdr (
        {Keys.SQUARE_ID} int,
        {Keys.TIME_INTERVAL} int,
        {Keys.COUNTRY_CODE} int,
        {Keys.SMS_IN} real,
        {Keys.SMS_OUT} real,
        {Keys.CALL_IN} real,
        {Keys.CALL_OUT} real,
        {Keys.INTERNET} real
    )''')
    con.commit()

    for input_file in listdir(input_dir):
        df = pd.read_csv(join(input_dir, input_file), sep='\t', header=None, names=columns)
        df.to_sql('cdr', con, if_exists='append', index=False)
    con.close()


def merge_text_files(input_files: Iterable[str], output_file: str):
    """Merge multiple clear text files into a single output file."""
    
    # create empty file
    with open(output_file, 'w') as _:
        pass

    # append each input tsv to the output tsv
    for input_file in sorted(input_files):
        with open(input_file, 'r') as f_in:
            print(f'process {os.getpid()} - appending {input_file.split(os.sep)[-1]}')
            with open(output_file, 'a+') as f_out:
                f_out.write(f_in.read())


def merge_to_size(max_size: float, input_dir: str, output_dir: str) -> None:
    '''
    Parallel merge all clear text files in a directory into files of a specified file size in Gigabytes.
    
    max_size: float -- max file size in Gigabytes
    input_dir: str -- path to directory that contains all clear text files to be merged
    output_dir: str -- path to directory that is going to contain all merged files
    '''

    # get all input files and their size in Gigabytes
    input_files = [join(input_dir, f) for f in sorted(listdir(input_dir))]
    input_files = [(f, getsize(f) / (1024 * 1024 * 1024)) for f in input_files]
    num_cpus = mp.cpu_count()

    # generate lists of files that are <= max_size in total
    # greedy approach, unbalanced
    cpu_files = []
    current_size = 0
    builder = []
    for f, size in input_files:
        if (current_size + size > max_size):
            cpu_files.append(builder)
            builder = []
            current_size = 0
        builder.append(f)
        current_size += size
    cpu_files.append(builder)

    # <output_dir>/merged_0xzy.tsv starting at 1
    output_files =  [ 
        join(output_dir, 'merged_' + str(i).zfill(len(cpu_files)) + '.tsv')
        for i in range(1, len(cpu_files) + 1)
    ]
    
    # run on all available cores, 1 per output file
    with mp.Pool(num_cpus) as p:
        p.starmap(merge_text_files, zip(cpu_files, output_files))

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

def group_by_country_code(
        df: pd.DataFrame,
        groupby_cols: Iterable[str] = [Keys.SQUARE_ID, Keys.TIME_INTERVAL, Keys.COUNTRY_CODE],
        agg_cols: Iterable[str] = [Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT, Keys.INTERNET],
        agg_method: str = 'sum'
    ) -> pd.DataFrame:
    '''
    Group all rows by square_id, and time_interval then
    replace country codes with 0 (local), and 1 (foreign) and
    aggregate CDRs on rows grouped by country codes.
    '''

    # cannot group by non-existing key
    if Keys.COUNTRY_CODE not in df:
        return df

    # country code mapping
    # 0 -> local
    # 1 -> foreign
    def country_code_mapping(x):
        return 0 if x in [0, 39] else 1
    df[Keys.COUNTRY_CODE] = df[Keys.COUNTRY_CODE].apply(country_code_mapping)

    return groupby_agg(df, groupby_cols, agg_cols, agg_method)


def load_tsv(
        input_file: str = Paths.MERGED_RAW_F,
        columns: Iterable[str] = [
            Keys.SQUARE_ID, Keys.TIME_INTERVAL, Keys.COUNTRY_CODE,
            Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT,
            Keys.INTERNET
        ],
        parse_dates: bool = False
    ):
    '''
    Load tsv from an input file and put it into a dataframe.
    Takes a long time and a lot of memory.
    Using only specific columns is going to increase both the time and memory performance.
    
    This function should be extended with more preprocessing options
    and maybe load snapshots of preprocessed data if they are available to speed up loading times.
    
    input_file: str -- name of input file
    columns: Iterable[str] -- iterable containing the names of columns to load
    parse_dates: bool -- convert unix timestamp to pandas datetime
    '''
    
    # columns to be included in the final dataframe
    # should probably always contain square_id, time_interval,
    # and at least one CDR
    usecols = [TableData.INDICES[c] for c in columns]

    df = pd.read_csv(
        input_file,
        sep='\t', header=None, dtype=TableData.DTYPES, usecols=usecols
    )
    df.columns = list(columns)

    # convert times if desired and time_intervals were loaded
    if parse_dates and Keys.TIME_INTERVAL not in df.columns:
        raise AttributeError(f'You need to include "{Keys.TIME_INTERVAL}" in "columns" to convert it to datetime!')
    elif parse_dates:
        df[Keys.TIME_INTERVAL] = pd.to_datetime(df[Keys.TIME_INTERVAL], unit='ms', utc=True) \
                                .dt.tz_convert('CET') \
                                .dt.tz_localize(None)
    return df


def dump_pickle(df: pd.DataFrame, output_file: str) -> None:
    '''Saves a dataframe as a binary file'''
    df.to_pickle(path=output_file, compression=None)


def load_pickle(input_file: str) -> pd.DataFrame:
    '''Loads a binary file into a dataframe'''
    return pd.read_pickle(input_file, compression=None)


if __name__ == '__main__':

    def merge_all_raw_to_tsv():
        '''Merge all 61 raw data files into a single tsv file.'''
        merge_text_files(
            input_files = [join(Paths.RAW_DIR, f) for f in listdir(Paths.RAW_DIR)],
            output_file = Paths.MERGED_RAW_F
        )
    # merge_all_raw_to_tsv()

    def group_country_codes_to_pickle(df=None):
        '''Reduce country codes to 0 -> local, 1 -> foreign and save as pickle'''
        if df is None:
            df = load_tsv(input_file = Paths.MERGED_RAW_F)
        df = group_by_country_code(df)
        dump_pickle(df, output_file= Paths.GROUPED_CC_F)
        return df
    # group_country_codes_to_pickle()

    def group_only_time_to_pickle(df=None):
        '''Reduce square id, time interval and country code to only time interval
           and save to pickle.'''
        if df is None:
            df = load_pickle(Paths.GROUPED_CC_F)
        df = groupby_agg(df, groupby_cols=[Keys.TIME_INTERVAL], agg_cols=[Keys.SMS_IN, Keys.SMS_OUT, Keys.CALL_IN, Keys.CALL_OUT, Keys.INTERNET], agg_method='sum')
        dump_pickle(df, Paths.GROUPED_TIME_F)
        return df
    # group_only_time_to_pickle()
