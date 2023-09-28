BEGIN;
CREATE TABLE IF NOT EXISTS raw_data (
    sq_id INTEGER, 
    time TEXT, 
    cc INTEGER, 
    sms_in REAL, 
    sms_out REAL, 
    call_in REAL, 
    call_out REAL, 
    internet REAL,
    PRIMARY KEY (sq_id, time, cc));
CREATE TABLE IF NOT EXISTS zones (
    zone INTEGER, 
    sq_id INTEGER,
    PRIMARY KEY (zone, sq_id));
COMMIT;