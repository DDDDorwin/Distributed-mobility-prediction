import pandas as pd
import sqlite3
import os
import sys
import csv

db_name = "raw_db"
db_path = "data/"
raw_path = "data/"

def insert_data_from_files(cursor):
    for file in os.listdir(raw_path):
        fn = os.fsdecode(file)
        #Add data only from .txt, .tsv or .csv files in directory
        if fn.endswith(".txt") or fn.endswith(".tsv") or fn.endswith(".csv"):
            with open(os.path.join(raw_path, fn), 'r') as data:
                dir = csv.reader(data, delimiter='\t')
                cursor.executemany("INSERT INTO raw_data (sq_id, time, cc, sms_in, sms_out, call_in, call_out, internet) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", dir)


#sq_id INTEGER, 
#time TEXT, 
#cc INTEGER, 
#sms_in REAL, 
#sms_out REAL, 
#call_in REAL, 
#call_out REAL, 
#internet REAL,

def list_db(cursor):
    cursor.execute("SELECT * FROM raw_data")
    print(cursor.fetchall())

def setup_db():
    #Delete old DB if exists
    if os.path.exists("%s%s.db" % (db_path, db_name)):
        os.remove("%s%s.db" % (db_path, db_name))

    db = sqlite3.connect("data/raw_db.db")
    cursor = db.cursor()

    #Setup DB tables
    with open("setup_db.sql") as file:
        script = file.read()
        cursor.executescript(script)

    #Populate DB with data from tsv
    insert_data_from_files(cursor)

    list_db(cursor)

    #Close DB connection
    cursor.close()
    db.close()





setup_db()




