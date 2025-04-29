import os
import random
import multiprocessing
from super_gradients.training import models
import glob
from collections import Counter
import logging
import sqlite3
import json
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from SPauth import hash_password, verify_password, generate_token, verify_token, is_token_blacklisted, SECRET_KEY
from SPencrypt import encrypt_data, decrypt_data
from SPsecurity import limiter, is_strong_password, sanitize_input, log_security_incident, is_valid_email
import datetime
import hashlib
import hmac

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'speedpark.db')
ML_JSON_PATH = os.path.join(os.path.dirname(__file__), 'parking_data.json')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_user(username: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

@app.route('/register', methods=['POST'])
@limiter.limit("10 per minute")
def register():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        username = sanitize_input(data.get("username"))
        password = data.get("password")
        email = sanitize_input(data.get("email"))
        if not all([username, password, email]):
            return jsonify({"error": "Missing required fields"}), 400
        if not is_valid_email(email):
            return jsonify({"error": "Invalid email format"}), 400
        if not is_strong_password(password):
            return jsonify({"error": "Weak password"}), 400
        hashed_password = hash_password(password)
        encrypted_email = encrypt_data(email)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                       (username, hashed_password, encrypted_email))
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully"}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Username already exists"}), 400
    except Exception as e:
        logging.error(f"Register error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        username = sanitize_input(data.get("username"))
        password = data.get("password")
        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and verify_password(password, user["password"]):
            token = generate_token(user_id=user["id"])
            return jsonify({"token": token})
        else:
            logging.warning(f"Failed login attempt from {request.remote_addr}")
            return jsonify({"error": "Invalid Credentials"}), 401
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# This method calculates the number of free and taken parking spaces in a 
# parking lot based on an image. It selects a random image from the folder
# representing a parking lot, which is named according to the lotID number.
#
# Boden Kahn
# 
# @param lotID  The id number of the lot to analyze.
# @return       The a list containing the number of free spaces in the image as
#               calculated by the machine learning model and the number of 
#               unavailable spaces as calculated by the model.
def getSpaceCount(lotID = 1):
    # Specify parameters
    MODEL_NAME = 'yolo_nas_l'  # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    CLASSES = ['free_parking_space', 'not_free_parking_space']
    NUM_CLASSES = len(CLASSES)

    # Load the model
    best_model = models.get(MODEL_NAME,
                            num_classes = NUM_CLASSES,
                            checkpoint_path = os.path.abspath("C:/SpeedParkModel/check_point/SpeedPark/RUN_20250427_082814_640062/ckpt_best.pth"))

    # Find available test images in the lot's folder
    directory = f"C:/tempTest/{lotID}/"
    all_images = glob.glob(os.path.join(directory, "**", "*.jpg"), recursive = True)

    if not all_images:
        print(f"No images found in {directory}")
        return -1

    # Pick a random valid image
    chosen_image = random.choice(all_images)
    print(f"Selected image: {chosen_image}")

    # Predict using the model
    result = best_model.predict(chosen_image, conf = 0.6)

    if not result:  # If no results, exit
        print("No predictions were made.")
        return -1

    # Get the labels from the prediction
    labels = result.prediction.labels

    # Count occurrences of each label
    count = Counter(labels)
    free_spaces = count.get(0, 0)
    not_free_spaces = count.get(1, 0)

    # Return the number of available and unavailable spaces
    return [free_spaces, not_free_spaces]

@app.route('/parking_availability', methods=['GET'])
@limiter.limit("30 per minute")
def parking_availability():
    try:
        logging.debug("Processing parking_availability")
        auth_header = request.headers.get('Authorization')
        logging.debug(f"Auth header: {auth_header}")
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.debug("Missing or invalid token")
            return jsonify({"error": "Missing or invalid token"}), 401
        token = auth_header.split(" ")[1]
        logging.debug(f"Token: {token}")
        if is_token_blacklisted(token):
            logging.debug("Token is blacklisted")
            return jsonify({"error": "Token is blacklisted"}), 401
        user_id = verify_token(token)
        logging.debug(f"User ID: {user_id}")
        if not user_id:
            logging.debug("Invalid or expired token")
            return jsonify({"error": "Invalid or expired token"}), 401
        lot_name = request.args.get('lot', 'Main Lot')
        # Read ML JSON file
        #try:
        #    with open(ML_JSON_PATH, 'r') as f:
        #        ml_data = json.load(f)
        #    logging.debug(f"ML data: {ml_data}")
        #    if ml_data.get('lot') != lot_name:
        #        logging.debug(f"Lot {lot_name} not found in ML data")
        #        return jsonify({"error": "Lot not found"}), 404
        #    spots = int(ml_data.get('available_spots', 5))
        #    free_spots = ml_data.get('free_spots', [])
        #except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        #    logging.error(f"ML file error: {str(e)}")
        #    return jsonify({"error": "Failed to fetch ML parking data"}), 500
        timestamp = datetime.datetime.utcnow().isoformat()
        spots = getSpaceCount(lot_name)
        free_spots = spots[0]
        not_free_spots = spots[1]
        data = {
            "lot_name": lot_name,
            #"available_spots": spots,
            "free_spots": free_spots,  # For app's visual display
            "not_free_spots": not_free_spots,
            "timestamp": timestamp
        }
        logging.debug(f"Data: {data}")
        signature = hmac.new(
            SECRET_KEY.encode(),
            str(data).encode(),
            hashlib.sha256
        ).hexdigest()
        logging.debug(f"Signature: {signature}")
        return jsonify({"data": data, "signature": signature})
    except Exception as e:
        logging.error(f"Parking availability error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.debug("Starting server initialization")
    try:
        if not os.path.exists(DB_PATH):
            logging.debug("Creating database at %s", DB_PATH)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE token_blacklist (
                    token TEXT PRIMARY KEY,
                    expiry DATETIME NOT NULL
                )
            """)
            conn.commit()
            logging.debug("Database created successfully")
        else:
            logging.debug("Database already exists at %s", DB_PATH)
    except Exception as e:
        logging.error("Failed to initialize database: %s", str(e))
        raise
    finally:
        if 'conn' in locals():
            conn.close()
    logging.debug("Starting Flask server")
    app.run(debug=True, use_reloader=False)
