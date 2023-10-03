import time
import pandas as pd


INPUT_F = '/home/tobi/projects/Project_CS_UserVsSpecific/data/benchmarks/'
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


def tsv():
    input_file = INPUT_F + 'bench.tsv'
    df = pd.read_csv(input_file, sep='\t', header=None, dtype=DTYPES)
    print(df.head(2))


def to_pickle():
    input_file = INPUT_F + 'bench.tsv'
    df = pd.read_csv(input_file, sep='\t', header=None, dtype=DTYPES)
    df.to_pickle(INPUT_F + 'bench.pickle', compression=None)


def pickle():
    input_file = INPUT_F + 'bench.pickle'
    df = pd.read_pickle(input_file, compression=None)
    print(df.head(2))


def to_feather():
    input_file = INPUT_F + 'bench.pickle'
    df = pd.read_pickle(input_file, compression=None)
    df.to_feather(INPUT_F + 'bench.feather')


def feather():
    input_file = INPUT_F + 'bench.feather'
    df = pd.read_feather(input_file)
    print(df.head(2))


def to_orc():
    input_file = INPUT_F + 'bench.pickle'
    df = pd.read_pickle(input_file, compression=None)
    df.to_orc(INPUT_F + 'bench.orc')


def orc():
    input_file = INPUT_F + 'bench.orc'
    df = pd.read_orc(input_file)
    print(df.head(2))
    df.read_numpy


def to_bz2():
    input_file = INPUT_F + 'bench.pickle'
    df = pd.read_pickle(input_file, compression=None)
    df.to_pickle(INPUT_F + 'bench.bz2', compression='bz2')


def bz2():
    input_file = INPUT_F + 'bench.bz2'
    df = pd.read_pickle(input_file, compression='bz2')
    print(df.head(2))


def to_parquet():
    input_file = INPUT_F + 'bench.pickle'
    df = pd.read_pickle(input_file, compression=None)
    df.to_parquet(INPUT_F + 'bench.parquet', compression=None, index=False)


def parquet():
    input_file = INPUT_F + 'bench.parquet'
    df = pd.read_parquet(input_file)
    print(df.head(2))

if __name__ == '__main__':
    start = time.perf_counter()
    parquet()
    end = time.perf_counter()
    with open('bench.txt', 'a+') as f:
        f.write(f'| parquet | mem | {(end - start):.2f} |\n')
    # to_pickle()
    # to_feather()
    # to_orc()
    # to_bz2()
    # to_parquet()
