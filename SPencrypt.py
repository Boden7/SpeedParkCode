import os
import logging
from cryptography.fernet import Fernet

SECRET_KEY = os.getenv("ENCRYPTION_KEY")
if not SECRET_KEY:
    SECRET_KEY = Fernet.generate_key()
    with open("encryption.key", "wb") as key_file:
        os.chmod("encryption.key", 0o600)
        key_file.write(SECRET_KEY)

cipher = Fernet(SECRET_KEY)

def encrypt_data(data: str) -> str:
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    try:
        return cipher.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        logging.error(f"Decryption failed: {str(e)}")
        raise ValueError("Unable to decrypt data")
