# check_db.py
import sqlite3
import os

db_path = os.path.join(r"C:\Users\patri\OneDrive\Desktop\Capstone\SP 2", "speedpark.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT username, email FROM users WHERE username='testuser'")
result = cursor.fetchone()
if result:
    print(f"Username: {result[0]}")
    print(f"Encrypted email: {result[1]}")
else:
    print("No user found")
conn.close()