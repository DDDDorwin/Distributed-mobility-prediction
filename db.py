import pandas as pd
import sqlite3
import os
import sys
import csv

db_name = "sqlite_db"
db_path = "data/"
raw_path = "data/raw_nov_10/"

class _DataBase:

    def __init__(self, setup_new):
        self.connection = sqlite3.connect("%s%s.db" % (db_path, db_name))
        self.cursor = self.connection.cursor()
        if setup_new == True:
            self.setup_new_db()
    
    def __del__(self):
        self.cursor.close()
        self.connection.close()

    def insert_data_from_files(self):
        for file in os.listdir(raw_path):
            fn = os.fsdecode(file)
            #Add data only from .txt, .tsv or .csv files in directory
            if fn.endswith(".txt") or fn.endswith(".tsv") or fn.endswith(".csv"):
                with open(os.path.join(raw_path, fn), 'r') as data:
                    dir = csv.reader(data, delimiter='\t')
                    self.cursor.executemany("INSERT INTO raw_data (sq_id, time, cc, sms_in, sms_out, call_in, call_out, internet) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", dir)
        self.connection.commit()

    def print_all(self):
        self.cursor.execute("SELECT * FROM raw_data")
        print(self.cursor.fetchall())

    def fetch_all(self):
        return pd.read_sql_query("SELECT * FROM raw_data", self.connection)

    def fetch_all_col(self, col):
        return pd.read_sql_query("SELECT sq_id, time, %s FROM raw_data" % (col), self.connection)

    def fetch_chunk(self, time_from, time_to):
        return pd.read_sql_query("SELECT * FROM raw_data WHERE time >= %s AND time <= %s" % (time_from, time_to), self.connection)

    def fetch_chunk_col(self, time_from, time_to, col):
        return pd.read_sql_query("SELECT sq_id, time, %s FROM raw_data WHERE time >= %s AND time <= %s" % (col, time_from, time_to), self.connection)

    def setup_new_db(self):
        #Delete old DB if exists
        if os.path.exists("%s%s.db" % (db_path, db_name)):
            os.remove("%s%s.db" % (db_path, db_name))

        self.connection = sqlite3.connect("%s%s.db" % (db_path, db_name))
        self.cursor = self.connection.cursor()

        #Setup DB tables
        with open("setup_db.sql") as file:
            script = file.read()
            self.cursor.executescript(script)

        #Populate DB with data from tsv
        self.insert_data_from_files()

db = _DataBase(False)
db.fetch_all()
del db

