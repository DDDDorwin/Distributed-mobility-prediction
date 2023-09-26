BEGIN;
CREATE TABLE IF NOT EXISTS test (id INTEGER, dt TEXT);
INSERT INTO test (id, dt) VALUES (1, 'hello');
INSERT INTO test (id, dt) VALUES (2, 'is it me');
INSERT INTO test (id, dt) VALUES (3, 'you are looking for');
COMMIT;