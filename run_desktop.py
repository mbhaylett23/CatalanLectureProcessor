"""Desktop entry point for Catalan Lecture Processor."""

import logging
import os
import sys
from html import escape

import gradio as gr
from fastapi import Form
from fastapi.responses import HTMLResponse

# Add project root to path so 'core' and 'ui' can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding - prevents UnicodeEncodeError with progress bars
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# Suppress HuggingFace tqdm download bars (they interfere with Gradio progress)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from core.auth import (  # noqa: E402
    get_auth_backend_name,
    get_login_message,
    has_auth_backend,
    is_firebase_auth_enabled,
    signup_user,
    verify_user,
)
from ui.app import create_app  # noqa: E402


def _signup_page(message: str = "", is_error: bool = False) -> str:
    message_html = ""
    if message:
        color = "#8b1e3f" if is_error else "#1f5d3d"
        background = "#fde7ee" if is_error else "#e7f6ec"
        message_html = (
            f"<div style='margin-bottom: 16px; padding: 12px; border-radius: 10px; "
            f"background: {background}; color: {color};'>{escape(message)}</div>"
        )

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Create Account</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, serif;
      background: linear-gradient(135deg, #f5efe6, #d6e4f0);
      color: #1f2933;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    .card {{
      width: min(460px, 100%);
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(31, 41, 51, 0.08);
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(31, 41, 51, 0.12);
      padding: 28px;
    }}
    h1 {{ margin-top: 0; margin-bottom: 8px; }}
    p {{ line-height: 1.5; }}
    label {{
      display: block;
      margin-top: 14px;
      margin-bottom: 6px;
      font-weight: 600;
    }}
    input {{
      width: 100%;
      box-sizing: border-box;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid #c9d4df;
      font-size: 15px;
    }}
    button {{
      margin-top: 18px;
      width: 100%;
      padding: 12px 14px;
      border: 0;
      border-radius: 10px;
      font-size: 16px;
      font-weight: 700;
      background: #2b6cb0;
      color: white;
      cursor: pointer;
    }}
    a {{ color: #2b6cb0; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>Create Account</h1>
    <p>Create a Firebase login for the lecture processor, then return to the login page.</p>
    {message_html}
    <form method="post" action="/signup">
      <label for="display_name">Display name</label>
      <input id="display_name" name="display_name" type="text" maxlength="120" placeholder="Jane Doe">

      <label for="email">Email</label>
      <input id="email" name="email" type="email" autocomplete="email" required>

      <label for="password">Password</label>
      <input id="password" name="password" type="password" minlength="6" autocomplete="new-password" required>

      <button type="submit">Create Account</button>
    </form>
    <p style="margin-top: 16px;"><a href="/">Back to login</a></p>
  </main>
</body>
</html>
"""


def _register_signup_routes(server_app):
    @server_app.get("/signup", response_class=HTMLResponse)
    async def signup_form():
        return HTMLResponse(_signup_page())

    @server_app.post("/signup", response_class=HTMLResponse)
    async def signup_submit(
        email: str = Form(...),
        password: str = Form(...),
        display_name: str = Form(""),
    ):
        try:
            signup_user(email.strip(), password, display_name.strip())
            return HTMLResponse(
                _signup_page(
                    "Account created. Return to the login page and sign in with your new credentials."
                )
            )
        except Exception as exc:
            return HTMLResponse(_signup_page(str(exc), is_error=True), status_code=400)


if __name__ == "__main__":
    app = create_app(mode="auto")

    auth_fn = verify_user if has_auth_backend() else None
    if auth_fn:
        logging.info("Authentication enabled (%s backend)", get_auth_backend_name())
    else:
        logging.warning("No auth backend found; running without login")

    server_app, _, _ = app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True,
        auth=auth_fn,
        auth_message=get_login_message(),
        prevent_thread_lock=True,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 960px !important; margin: auto; }
            #status-box textarea {
                font-size: 1.1em !important;
                font-weight: bold !important;
                background-color: #1a1a2e !important;
                border: 2px solid #e94560 !important;
                color: #eee !important;
            }
        """,
    )

    if is_firebase_auth_enabled():
        _register_signup_routes(server_app)

    app.block_thread()
