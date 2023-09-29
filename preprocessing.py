import os
from os import listdir
from os.path import join, getsize
import multiprocessing as mp

import pandas as pd


RAW_DIR = '/home/tobi/projects/Project_CS_UserVsSpecific/data/raw/'
PREPROCESSED_DIR = '/home/tobi/projects/Project_CS_UserVsSpecific/data/preprocessed/'
GROUPED_CC_DIR = '/home/tobi/projects/Project_CS_UserVsSpecific/data/preprocessed/grouped_country_codes/'
MERGED_TO_SIZE_DIR = '/home/tobi/projects/Project_CS_UserVsSpecific/data/preprocessed/merged_to_size/'


DTYPES = {
    'square_id': 'int16',
    'time_interval': 'int64',
    'country_code': 'int8',
    'sms_in': 'float64',
    'sms_out': 'float64',
    'call_in': 'float64',
    'call_out': 'float64',
    'internet_traffic': 'float64'
}


def raw_tsv_to_sqlite(input_dir=RAW_DIR, output_dir=PREPROCESSED_DIR, output_file='raw.sqlite3'):
    '''Merge all tsv files into a single sqlite3 database file.'''
    
    import sqlite3

    # read tsv
    columns = ['square_id', 'time_interval', 'country_code', 'sms_in',
               'sms_out', 'call_in', 'call_out', 'internet_traffic']
    
    # create data base file if it doesn't exist yet
    con = sqlite3.connect(join(output_dir, output_file))
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS cdr (
        square_id int,
        time_interval int,
        country_code int,
        sms_in real,
        sms_out real,
        call_in real,
        call_out real,
        internet_traffic real
    )''')
    con.commit()

    for input_file in listdir(input_dir):
        df = pd.read_csv(join(input_dir, input_file), sep='\t', header=None, names=columns)
        df.to_sql('cdr', con, if_exists='append', index=False)
    con.close()


def merge(input_files, output_file):
    """Merge input files into a single output file."""
    
    # create empty file
    with open(output_file, 'w') as _:
        pass

    # append each input tsv to the output tsv
    for input_file in sorted(input_files):
        with open(input_file, 'r') as f_in:
            print(f'process {os.getpid()} - appending {input_file.split(os.sep)[-1]}')
            with open(output_file, 'a+') as f_out:
                f_out.write(f_in.read())


def merge_to_size(max_size, input_dir, output_dir):
    '''
    Parallel merge all files in a directory into files of a specified file size in Gigabytes.
    
    max_size: float -- max file size in Gigabytes
    input_dir: str -- path to directory that contains all files to be merged
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
        p.starmap(merge, zip(cpu_files, output_files))

def _group_by_country_code(input_files, output_dir=GROUPED_CC_DIR):
    '''
    Worker function for multiprocessing.

    Group all rows by square_id, and time_interval then
    replace country codes with 0 (local), and 1 (foreign) and
    aggregate CDRs on rows grouped by country codes.
    '''

    for input_file in input_files:
        print(f'process {os.getpid()} - grouping country codes of {input_file.split(os.sep)[-1]}')
        df = pd.read_csv(input_file, delimiter='\t', header=None, names=list(DTYPES.keys()), dtype=DTYPES)

        # country code mapping
        # 0 -> local
        # 1 -> foreign
        def country_code_mapping(x):
            return 0 if x in [0, 39] else 1
        df['country_code'] = df['country_code'].apply(country_code_mapping)

        # aggregate rows with same id, time, and country code
        # this will automatically fill some NaNs with zeroes
        grouped_df = df.groupby(['square_id', 'time_interval', 'country_code']).agg({
            'sms_in': 'sum',
            'sms_out': 'sum',
            'call_in': 'sum',
            'call_out': 'sum',
            'internet_traffic': 'sum'
        }).reset_index()

        # save to disk
        grouped_df.to_csv(join(output_dir, input_file), sep='\t', header=False, index=False)


def group_by_country_codes(input_dir=RAW_DIR):
    '''
    Parallel group all rows by square_id, and time_interval then
    replace country codes with 0 (local), and 1 (foreign) and
    aggregate CDRs on rows grouped by country codes.
    '''

    # get all input files from directory and available cores
    input_files = [join(input_dir, f) for f in sorted(listdir(input_dir))]
    num_cpus = mp.cpu_count()

    # split input files up between cores
    # assuming operation is independent
    cpu_files = [
        input_files[i::num_cpus]
        for i in range(num_cpus)
    ]

    # run on all available cores
    with mp.Pool(num_cpus) as p:
        p.map(_group_by_country_code, cpu_files)


def load_data(
        columns = [
            'square_id',
            'time_interval',
            'country_code',
            'sms_in', 'sms_out',
            'call_in', 'call_out',
            'internet_traffic'
        ],
        parse_dates = False,
        input_dir = PREPROCESSED_DIR,
        input_file ='raw_merged.tsv'
    ):
    '''
    Load tsv from an input file and put it into a dataframe.
    Takes a long time and a lot of memory.
    Using only specific columns is going to increase both the time and memory performance.
    
    This function should be extended with more preprocessing options
    and maybe load snapshots of preprocessed data if they are available to speed up loading times.

    columns: Iterable[str] -- iterable containing the names of columns to load
    parse_dates: bool -- convert unix timestamp to pandas datetime
    input_dir: str -- directory of input file
    input_file: str -- name of input file
    '''

    column_indices = {
        'square_id': 0,
        'time_interval': 1,
        'country_code': 2,
        'sms_in': 3,              
        'sms_out': 4,
        'call_in': 5,
        'call_out': 6,
        'internet_traffic': 7,
    }
    
    # columns to be included in the final dataframe
    # should probably always contain square_id, time_interval,
    # and at least one CDR
    usecols = [column_indices[c] for c in columns]

    df = pd.read_csv(
        join(input_dir, input_file),
        sep='\t', header=None, dtype=DTYPES, usecols=usecols
    )
    df.columns = columns

    # convert times if desired and time_intervals were loaded
    if parse_dates and 'time_interval' not in df.columns:
        raise AttributeError('You need to include "time_interval" in "columns" to convert it to datetime!')
    elif parse_dates:
        df['time_interval'] = pd.to_datetime(df['time_interval'], unit='ms', utc=True) \
                                .dt.tz_convert('CET') \
                                .dt.tz_localize(None)
        
    return df
