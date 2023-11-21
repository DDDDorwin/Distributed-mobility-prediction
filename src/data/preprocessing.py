import os
import torch
import pandas as pd
from typing import Iterable, List
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
