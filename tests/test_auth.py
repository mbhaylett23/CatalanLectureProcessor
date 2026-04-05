"""Tests for hybrid auth helpers."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import auth as auth_module


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset_auth_state(monkeypatch):
    for env_var in (
        "FIREBASE_WEB_API_KEY",
        "FIREBASE_API_KEY",
        "FIREBASE_SERVICE_ACCOUNT_FILE",
        "FIREBASE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(env_var, raising=False)

    auth_module._FIREBASE_USER_CACHE.clear()
    auth_module._get_firebase_admin_auth.cache_clear()


def test_json_auth_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_module, "AUTH_DIR", str(tmp_path))
    monkeypatch.setattr(auth_module, "USERS_FILE", str(tmp_path / "users.json"))

    assert auth_module.add_user("alice", "secret123", "Alice Example") is True
    assert auth_module.verify_user("alice", "secret123") is True
    assert auth_module.verify_user("alice", "wrong") is False
    assert auth_module.get_user_storage_id("alice") == "alice"
    assert auth_module.list_users() == [{"username": "alice", "name": "Alice Example"}]


def test_firebase_verify_user_caches_uid(monkeypatch):
    monkeypatch.setenv("FIREBASE_WEB_API_KEY", "test-key")

    def fake_post(url, json, timeout):
        assert "accounts:signInWithPassword" in url
        assert json["email"] == "student@example.com"
        return _FakeResponse(
            200,
            {
                "localId": "firebase-uid-123",
                "displayName": "Student Example",
            },
        )

    monkeypatch.setattr(auth_module.requests, "post", fake_post)

    assert auth_module.verify_user("student@example.com", "secret123") is True
    assert auth_module.get_user_storage_id("student@example.com") == "firebase-uid-123"


def test_signup_user_sets_display_name(monkeypatch):
    monkeypatch.setenv("FIREBASE_WEB_API_KEY", "test-key")
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        if "accounts:signUp" in url:
            return _FakeResponse(
                200,
                {
                    "idToken": "token-123",
                    "localId": "firebase-uid-456",
                },
            )
        if "accounts:update" in url:
            assert json["displayName"] == "Student Name"
            return _FakeResponse(
                200,
                {
                    "displayName": "Student Name",
                },
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(auth_module.requests, "post", fake_post)

    user = auth_module.signup_user(
        "student@example.com",
        "secret123",
        "Student Name",
    )

    assert user["uid"] == "firebase-uid-456"
    assert user["name"] == "Student Name"
    assert len(calls) == 2
