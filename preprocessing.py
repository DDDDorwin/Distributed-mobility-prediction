from os import listdir
from os.path import join

import pandas as pd


RAW_DIR = '/home/tobi/projects/Project_CS_UserVsSpecific/data/raw/'
PREPROCESSED_DIR = '/home/tobi/projects/Project_CS_UserVsSpecific/data/preprocessed/'


def merge_raw_data(input_dir=RAW_DIR, output_dir=PREPROCESSED_DIR, output_file='raw_merged.tsv'):
    """Merge all raw tsv files into a single tsv file."""
    
    # create empty file
    with open(join(output_dir, output_file), 'w') as f:
        pass

    # append each input tsv to the output tsv
    for input_file in listdir(input_dir):
        with open(join(input_dir, input_file), 'r') as f_in:
            with open(join(output_dir, output_file), 'a+') as f_out:
                f_out.write(f_in.read())

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

def load_data(
        # columns = [
        #     'square_id',
        #     'time_interval',
        #     'country_code',
        #     'sms_in', 'sms_out',
        #     'call_in', 'call_out',
        #     'internet_traffic'
        # ],
        columns = [
                    'time_interval',
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

    dtypes = {
        'square_id': 'int16',
        'time_interval': 'int16',
        'country_code': 'int8',
        'sms_in': 'float32',              
        'sms_out': 'float32',
        'call_in': 'float32',
        'call_out': 'float32',
        'internet_traffic': 'float32'
    }

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
    
    usecols = [column_indices[c] for c in columns]

    # assign column names while reading the inputs
    df = pd.read_csv(
        join(input_dir, input_file),
        sep='\t', header=None, dtype=dtypes, names=['Square_id', 'Time_interval', 'Country_code', 'SMS_in',
           'SMS_out', 'Call_in','Call_out', 'Internet_traffic'], usecols=usecols
    )

    if parse_dates:
        df['Time_interval'] = pd.to_datetime(df['Time_interval'], unit='ms', utc=True) \
                                .dt.tz_convert('CET') \
                                .dt.tz_localize(None)
        
    return df
