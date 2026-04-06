"""Authentication helpers for the Catalan Lecture Processor."""

import hashlib
import json
import logging
import os
import secrets
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)

AUTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "auth")
USERS_FILE = os.path.join(AUTH_DIR, "users.json")

_FIREBASE_USER_CACHE: dict[str, dict[str, str]] = {}
_FIREBASE_ADMIN_ERROR_LOGGED = False


class FirebaseAuthError(RuntimeError):
    """Raised when Firebase Authentication returns an error."""


def _load_users() -> dict:
    """Load users from JSON file."""
    if not os.path.isfile(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    """Write users dict to JSON file."""
    os.makedirs(AUTH_DIR, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-SHA256. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations=100_000)
    return pw_hash.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Check a password against stored hash + salt."""
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations=100_000)
    return secrets.compare_digest(pw_hash.hex(), stored_hash)


def _resolve_firebase_web_api_key() -> str | None:
    for env_var in ("FIREBASE_WEB_API_KEY", "FIREBASE_API_KEY"):
        value = os.environ.get(env_var)
        if value and value.strip():
            return value.strip()
    return None


def is_firebase_auth_enabled() -> bool:
    """Return True when Firebase email/password auth is configured."""
    return _resolve_firebase_web_api_key() is not None


def has_auth_backend() -> bool:
    """Return True when either Firebase or JSON auth is available."""
    return is_firebase_auth_enabled() or os.path.isfile(USERS_FILE)


def get_auth_backend_name() -> str:
    """Return the active auth backend name."""
    if is_firebase_auth_enabled():
        return "firebase"
    if os.path.isfile(USERS_FILE):
        return "json"
    return "none"


def get_login_message() -> str | None:
    """Optional HTML shown on the Gradio login screen."""
    if is_firebase_auth_enabled():
        return "Need an account? Click the button below."
    return None


def _firebase_endpoint(action: str) -> str:
    api_key = _resolve_firebase_web_api_key()
    if not api_key:
        raise FirebaseAuthError("Firebase web API key is not configured")
    return f"https://identitytoolkit.googleapis.com/v1/accounts:{action}?key={api_key}"


def _firebase_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "Unknown Firebase Authentication error"

    message = payload.get("error", {}).get("message")
    friendly_messages = {
        "EMAIL_EXISTS": "An account with that email already exists.",
        "EMAIL_NOT_FOUND": "No account was found for that email address.",
        "INVALID_PASSWORD": "The password is incorrect.",
        "INVALID_LOGIN_CREDENTIALS": "The email or password is incorrect.",
        "WEAK_PASSWORD : Password should be at least 6 characters": (
            "Password must be at least 6 characters long."
        ),
        "MISSING_PASSWORD": "Password is required.",
        "INVALID_EMAIL": "Please enter a valid email address.",
    }
    return friendly_messages.get(message, message or "Unknown Firebase Authentication error")


def _firebase_request(action: str, payload: dict) -> dict:
    response = requests.post(
        _firebase_endpoint(action),
        json=payload,
        timeout=15,
    )
    if response.status_code >= 400:
        raise FirebaseAuthError(_firebase_error_message(response))
    return response.json()


def _cache_firebase_user(email: str, uid: str | None = None, name: str | None = None):
    _FIREBASE_USER_CACHE[email] = {
        "uid": uid or "",
        "name": name or email,
    }


def _resolve_service_account_source() -> tuple[str | None, dict | None]:
    raw_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if raw_json and raw_json.strip():
        return None, json.loads(raw_json)

    for env_var in ("FIREBASE_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        file_path = os.environ.get(env_var)
        if file_path and file_path.strip():
            return file_path.strip(), None

    return None, None


@lru_cache(maxsize=1)
def _get_firebase_admin_auth():
    global _FIREBASE_ADMIN_ERROR_LOGGED

    try:
        import firebase_admin
        from firebase_admin import auth as admin_auth
        from firebase_admin import credentials
    except ImportError:
        if is_firebase_auth_enabled() and not _FIREBASE_ADMIN_ERROR_LOGGED:
            logger.warning(
                "firebase-admin is not installed; admin operations and UID lookups may be limited"
            )
            _FIREBASE_ADMIN_ERROR_LOGGED = True
        return None

    if firebase_admin._apps:
        return admin_auth

    file_path, raw_json = _resolve_service_account_source()
    if raw_json:
        cred = credentials.Certificate(raw_json)
    elif file_path:
        cred = credentials.Certificate(file_path)
    else:
        return None

    try:
        firebase_admin.initialize_app(cred)
        return admin_auth
    except Exception as exc:
        if not _FIREBASE_ADMIN_ERROR_LOGGED:
            logger.warning("Firebase Admin SDK initialization failed: %s", exc)
            _FIREBASE_ADMIN_ERROR_LOGGED = True
        return None


def get_user_storage_id(username: str) -> str:
    """Return the folder-safe user identifier for storing outputs."""
    if not username:
        return ""

    if not is_firebase_auth_enabled():
        return username

    cached = _FIREBASE_USER_CACHE.get(username)
    if cached and cached.get("uid"):
        return cached["uid"]

    admin_auth = _get_firebase_admin_auth()
    if admin_auth is not None:
        try:
            user = admin_auth.get_user_by_email(username)
            _cache_firebase_user(username, uid=user.uid, name=user.display_name)
            return user.uid
        except Exception as exc:
            logger.warning("Firebase UID lookup failed for %s: %s", username, exc)

    return username.replace("@", "_at_").replace("/", "_")


def verify_user(username: str, password: str) -> bool:
    """Validate login credentials for Gradio auth."""
    if is_firebase_auth_enabled():
        try:
            data = _firebase_request(
                "signInWithPassword",
                {
                    "email": username,
                    "password": password,
                    "returnSecureToken": True,
                },
            )
            _cache_firebase_user(
                username,
                uid=data.get("localId"),
                name=data.get("displayName") or username,
            )
            return True
        except FirebaseAuthError as exc:
            logger.info("Firebase login rejected for %s: %s", username, exc)
            return False
        except Exception as exc:
            logger.warning("Firebase login failed for %s: %s", username, exc)
            return False

    users = _load_users()
    user = users.get(username)
    if not user:
        return False
    return verify_password(password, user["password_hash"], user["salt"])


def signup_user(username: str, password: str, name: str = "") -> dict:
    """Create a self-service user account."""
    if is_firebase_auth_enabled():
        data = _firebase_request(
            "signUp",
            {
                "email": username,
                "password": password,
                "returnSecureToken": True,
            },
        )
        uid = data.get("localId")
        display_name = name.strip()
        if display_name:
            try:
                update = _firebase_request(
                    "update",
                    {
                        "idToken": data["idToken"],
                        "displayName": display_name,
                        "returnSecureToken": True,
                    },
                )
                display_name = update.get("displayName") or display_name
            except FirebaseAuthError as exc:
                logger.warning("Firebase display name update failed for %s: %s", username, exc)
        else:
            display_name = username

        _cache_firebase_user(username, uid=uid, name=display_name)
        return {
            "username": username,
            "uid": uid,
            "name": display_name,
        }

    if not add_user(username, password, name):
        raise FirebaseAuthError(f"User '{username}' already exists")
    return {
        "username": username,
        "uid": username,
        "name": name or username,
    }


def add_user(username: str, password: str, name: str = "") -> bool:
    """Create a new user via Firebase Admin when available, else JSON fallback."""
    if is_firebase_auth_enabled():
        admin_auth = _get_firebase_admin_auth()
        if admin_auth is None:
            raise FirebaseAuthError(
                "Firebase Admin SDK is not configured. Set FIREBASE_SERVICE_ACCOUNT_FILE "
                "or FIREBASE_SERVICE_ACCOUNT_JSON."
            )
        try:
            user = admin_auth.create_user(
                email=username,
                password=password,
                display_name=name or None,
            )
            _cache_firebase_user(username, uid=user.uid, name=user.display_name or username)
            logger.info("Firebase user created: %s", username)
            return True
        except Exception as exc:
            logger.info("Firebase user creation failed for %s: %s", username, exc)
            return False

    users = _load_users()
    if username in users:
        return False
    pw_hash, salt = hash_password(password)
    users[username] = {
        "name": name or username,
        "password_hash": pw_hash,
        "salt": salt,
    }
    _save_users(users)
    logger.info("User created: %s", username)
    return True


def remove_user(username: str) -> bool:
    """Delete a user via Firebase Admin when available, else JSON fallback."""
    if is_firebase_auth_enabled():
        admin_auth = _get_firebase_admin_auth()
        if admin_auth is None:
            raise FirebaseAuthError(
                "Firebase Admin SDK is not configured. Set FIREBASE_SERVICE_ACCOUNT_FILE "
                "or FIREBASE_SERVICE_ACCOUNT_JSON."
            )
        try:
            user = admin_auth.get_user_by_email(username)
            admin_auth.delete_user(user.uid)
            _FIREBASE_USER_CACHE.pop(username, None)
            logger.info("Firebase user removed: %s", username)
            return True
        except Exception as exc:
            logger.info("Firebase user removal failed for %s: %s", username, exc)
            return False

    users = _load_users()
    if username not in users:
        return False
    del users[username]
    _save_users(users)
    logger.info("User removed: %s", username)
    return True


def reset_password(username: str, password: str) -> bool:
    """Reset a user's password."""
    if is_firebase_auth_enabled():
        admin_auth = _get_firebase_admin_auth()
        if admin_auth is None:
            raise FirebaseAuthError(
                "Firebase Admin SDK is not configured. Set FIREBASE_SERVICE_ACCOUNT_FILE "
                "or FIREBASE_SERVICE_ACCOUNT_JSON."
            )
        try:
            user = admin_auth.get_user_by_email(username)
            admin_auth.update_user(user.uid, password=password)
            logger.info("Firebase password reset for: %s", username)
            return True
        except Exception as exc:
            logger.info("Firebase password reset failed for %s: %s", username, exc)
            return False

    users = _load_users()
    if username not in users:
        return False
    pw_hash, salt = hash_password(password)
    users[username]["password_hash"] = pw_hash
    users[username]["salt"] = salt
    _save_users(users)
    logger.info("Password reset for: %s", username)
    return True


def list_users() -> list[dict]:
    """Return a list of users without secrets."""
    if is_firebase_auth_enabled():
        admin_auth = _get_firebase_admin_auth()
        if admin_auth is None:
            raise FirebaseAuthError(
                "Firebase Admin SDK is not configured. Set FIREBASE_SERVICE_ACCOUNT_FILE "
                "or FIREBASE_SERVICE_ACCOUNT_JSON."
            )
        users = []
        for user in admin_auth.list_users().iterate_all():
            users.append(
                {
                    "username": user.email or user.uid,
                    "name": user.display_name or user.email or user.uid,
                    "uid": user.uid,
                }
            )
        users.sort(key=lambda item: item["username"])
        return users

    users = _load_users()
    return [
        {"username": uname, "name": data.get("name", uname)}
        for uname, data in users.items()
    ]
