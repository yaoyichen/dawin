import sqlite3

conn = sqlite3.connect("../data/stock.db")

cursor = conn.cursor()

string1 = """CREATE TABLE contacts (
	contact_id INTEGER PRIMARY KEY,
	first_name TEXT NOT NULL,
	last_name TEXT NOT NULL,
	email TEXT NOT NULL UNIQUE,
	phone TEXT NOT NULL UNIQUE
);"""

# cursor.execute(string1)

task = (2, 'd','b','e', '190')
sql_str = '''INSERT INTO contacts (contact_id,first_name,last_name,email,phone) 
            VALUES (?,?,?,?,?);'''

print(sql_str)
cursor.execute(sql_str, task)
conn.commit()


conn.close()