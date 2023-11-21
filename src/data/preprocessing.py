import os
import torch
import pandas as pd
from typing import TypeVar
from constants import *
from os.path import join
from processing import nor

def del_f(destroy_dir):
    for file in os.listdir(destroy_dir):
        if os.fsdecode(file).endswith(".txt"):
            os.remove(join(destroy_dir, file))

def normalize(input_dir, output_dir, destroy_old):
    if destroy_old:
        del_f(output_dir)

    input_files = [
        join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")
    ]

  
    max_sms_in = 0
    max_sms_out = 0
    max_call_in = 0
    max_call_out = 0
    max_internet = 0

    for input_file in sorted(input_files):
        df = pd.read_csv(input_file, header=None, sep="\t", dtype=TableData.DTYPES)
        max_sms_in = max(max_sms_in, df[Keys.SMS_IN].max())
        max_sms_out = max(max_sms_out, df[Keys.SMS_OUT].max())
        max_call_in = max(max_call_in, df[Keys.CALL_IN].max())
        max_call_out = max(max_call_out, df[Keys.CALL_OUT].max())
        max_internet = max(max_internet, df[Keys.INTERNET].max())


        
       
        print("Successfully normalized %s" % (input_file))
    
    print(max_call_in)
    print(max_sms_out)
    print(max_call_in)
    print(max_call_out)
    print(max_internet)
    print("Finnished normalization")
