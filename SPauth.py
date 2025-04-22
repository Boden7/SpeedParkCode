import os
import bcrypt
import jwt
import datetime
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise EnvironmentError("JWT_SECRET_KEY environment variable must be set")

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def generate_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str):
    try:
        logging.debug(f"Verifying token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        logging.debug(f"Payload: {payload}")
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        logging.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logging.warning(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in verify_token: {str(e)}")
        return None

def blacklist_token(token: str):
    try:
        conn = sqlite3.connect("speedpark.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO token_blacklist (token, expiry) VALUES (?, ?)",
                      (token, datetime.datetime.utcnow() + datetime.timedelta(hours=2)))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error blacklisting token: {str(e)}")

def is_token_blacklisted(token: str) -> bool:
    try:
        conn = sqlite3.connect("speedpark.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM token_blacklist WHERE token = ?", (token,))
        result = cursor.fetchone()
        conn.close()
        return bool(result)
    except Exception as e:
        logging.error(f"Error checking blacklist: {str(e)}")
        return False

def get_user(username: str):
    try:
        conn = sqlite3.connect("speedpark.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        if not user:
            return None
        return {"id": user[0], "password": user[1]}
    except Exception as e:
        logging.error(f"Error getting user: {str(e)}")
        return None
