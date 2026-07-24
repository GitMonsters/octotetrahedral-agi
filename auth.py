import hashlib
import secrets
import json
from pathlib import Path
from datetime import datetime

API_KEYS_FILE = Path.home() / ".octotetrahedral" / "api_keys.json"

def generate_api_key(label="default"):
    """Generate a secure random API key"""
    key = secrets.token_urlsafe(32)
    save_api_key(label, key)
    return key

def hash_key(key):
    """Hash API key for secure storage"""
    return hashlib.sha256(key.encode()).hexdigest()

def save_api_key(label, key):
    """Save API key to file"""
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    keys = {}
    if API_KEYS_FILE.exists():
        with open(API_KEYS_FILE) as f:
            keys = json.load(f)
    
    keys[label] = {
        "hash": hash_key(key),
        "created": datetime.now().isoformat(),
        "requests": 0
    }
    
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def validate_api_key(key):
    """Validate API key"""
    if not API_KEYS_FILE.exists():
        return False
    
    try:
        with open(API_KEYS_FILE) as f:
            keys = json.load(f)
        
        key_hash = hash_key(key)
        for label, data in keys.items():
            if data["hash"] == key_hash:
                data["requests"] += 1
                with open(API_KEYS_FILE, "w") as f:
                    json.dump(keys, f, indent=2)
                return True
    except Exception:
        return False
    
    return False
