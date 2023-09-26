import sqlite3

db = sqlite3.connect("data/db.db")
cursor = db.cursor()

with open("setup_db.sql") as file:
    script = file.read()

cursor.executescript(script)

for row in cursor.execute("SELECT * FROM test"):
    print(row)

cursor.close()
db.close()

