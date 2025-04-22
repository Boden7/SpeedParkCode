import re
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging.handlers

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "10 per minute"],
    storage_uri="memory://"
)

logger = logging.getLogger('security')
handler = logging.handlers.RotatingFileHandler('security.log', maxBytes=10*1024*1024, backupCount=5)
logger.addHandler(handler)

def is_strong_password(password: str) -> bool:
    return bool(re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$', password))

def sanitize_input(user_input: str, allow_apostrophe: bool = False) -> str:
    pattern = r'[^\w\s@.\'-]' if allow_apostrophe else r'[^\w\s@.-]'
    return re.sub(pattern, '', user_input)

def log_security_incident(username: str, event: str):
    logger.warning(f"User: {sanitize_input(username)}, Event: {event}")

def is_valid_email(email: str) -> bool:
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email))
