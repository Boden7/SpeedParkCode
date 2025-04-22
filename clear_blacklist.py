# clear_blacklist.py
import sqlite3
db_path = r"C:\Users\patri\OneDrive\Desktop\Capstone\SP 2\speedpark.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("DELETE FROM token_blacklist")
conn.commit()
conn.close()