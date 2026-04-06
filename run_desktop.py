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

from core.pipeline import OUTPUT_DIR  # noqa: E402
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
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
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



def _custom_login_page() -> str:
    """Custom login page with a visible signup link."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Login</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: linear-gradient(135deg, #f5efe6, #d6e4f0);
      color: #1f2933;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .card {
      width: min(460px, 100%);
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(31, 41, 51, 0.08);
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(31, 41, 51, 0.12);
      padding: 28px;
    }
    h1 { margin-top: 0; margin-bottom: 8px; }
    p { line-height: 1.5; }
    label {
      display: block;
      margin-top: 14px;
      margin-bottom: 6px;
      font-weight: 600;
    }
    input {
      width: 100%;
      box-sizing: border-box;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid #c9d4df;
      font-size: 15px;
    }
    button {
      margin-top: 18px;
      width: 100%;
      padding: 14px;
      background: #7c3aed;
      color: #fff;
      border: none;
      border-radius: 10px;
      font-size: 16px;
      cursor: pointer;
    }
    button:hover { background: #6d28d9; }
    .signup-link {
      margin-top: 16px;
      text-align: center;
    }
    .signup-link a {
      color: #7c3aed;
      text-decoration: underline;
      font-weight: bold;
    }
    .error {
      margin-bottom: 16px;
      padding: 12px;
      border-radius: 10px;
      background: #fde7ee;
      color: #8b1e3f;
      display: none;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Login</h1>
    <p>Sign in to access the Catalan Lecture Processor.</p>
    <div class="error" id="error-msg"></div>
    <form id="login-form">
      <label for="username">Email</label>
      <input type="email" id="username" name="username" required>
      <label for="password">Password</label>
      <input type="password" id="password" name="password" required>
      <button type="submit">Login</button>
    </form>
    <div class="signup-link">
      <p>Need an account? <a href="/signup">Create one</a></p>
    </div>
  </div>
  <script>
    document.getElementById('login-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const form = e.target;
      const data = new URLSearchParams(new FormData(form));
      const resp = await fetch('/login', {method: 'POST', body: data,
                                          headers: {'Content-Type': 'application/x-www-form-urlencoded'}});
      if (resp.ok) {
        window.location.href = '/';
      } else {
        const errDiv = document.getElementById('error-msg');
        errDiv.textContent = 'Invalid email or password.';
        errDiv.style.display = 'block';
      }
    });
  </script>
</body>
</html>
"""


def _register_login_and_signup_routes(server_app):
    """Register custom /welcome login page and redirect / for unauthenticated users."""
    from starlette.routing import Route
    from starlette.responses import RedirectResponse, Response as StarletteResponse

    @server_app.get("/welcome", response_class=HTMLResponse)
    async def custom_login():
        return HTMLResponse(_custom_login_page())

    # Serve a JS file that adds a signup button to Gradio's login page
    @server_app.get("/signup-btn.js")
    async def signup_btn_js():
        from starlette.responses import Response
        return Response(content="""
(function addSignupButton(){
    // Only inject on the login page, not inside the app
    function isLoginPage(){
        return document.querySelector('input[type=password]') &&
               !document.querySelector('.gradio-container .app');
    }
    function tryInject(){
        if(!isLoginPage()) return;
        var btn=document.querySelector('button.lg')||document.querySelector('form button');
        if(!btn){setTimeout(tryInject,200);return;}
        if(document.getElementById('signup-btn'))return;
        var a=document.createElement('a');
        a.id='signup-btn';a.href='/signup';a.textContent='Create Account';
        a.style.cssText='display:block;text-align:center;margin-top:12px;padding:12px;background:#4f46e5;color:white;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px;';
        btn.parentNode.insertBefore(a,btn.nextSibling);
    }
    setTimeout(tryInject,800);
})();
""", media_type="application/javascript")

    # Wrap the entire ASGI app to inject the script tag into / responses
    import functools

    _orig_asgi = server_app.build_middleware_stack
    _ghost_loader = b"""<style>
#ghost-loader{position:fixed;top:0;left:0;width:100%;height:100%;background:#0b0f19;z-index:9999;display:flex;justify-content:center;align-items:center;}
#ghost-loader .card{width:min(400px,90%);background:#1a1e2e;border-radius:18px;padding:32px;animation:pulse 1.5s ease-in-out infinite;}
#ghost-loader .line{height:14px;background:#2a2e3e;border-radius:7px;margin-bottom:16px;}
#ghost-loader .line.short{width:60%;}
#ghost-loader .line.medium{width:80%;}
#ghost-loader .input-ghost{height:44px;background:#2a2e3e;border-radius:10px;margin-bottom:14px;}
#ghost-loader .btn-ghost{height:48px;background:#3a3080;border-radius:10px;margin-top:8px;}
@keyframes pulse{0%,100%{opacity:0.6;}50%{opacity:1;}}
</style>
<div id="ghost-loader">
  <div class="card">
    <div class="line short"></div>
    <div class="line medium" style="margin-bottom:24px;height:10px;"></div>
    <div class="line short" style="height:10px;width:40%;"></div>
    <div class="input-ghost"></div>
    <div class="line short" style="height:10px;width:40%;"></div>
    <div class="input-ghost"></div>
    <div class="btn-ghost"></div>
  </div>
</div>
<script>
(function hideGhost(){
    function check(){
        if(document.querySelector('button.lg')||document.querySelector('form button')||document.querySelector('.gradio-container')||document.querySelector('gradio-app')){
            var g=document.getElementById('ghost-loader');
            if(g)g.remove();
        }else{setTimeout(check,100);}
    }
    setTimeout(check,500);
})();
</script>"""
    _script_tag = b'<script src="/signup-btn.js"></script>' + _ghost_loader + b'</body>'

    @functools.wraps(_orig_asgi)
    def _patched_build():
        stack = _orig_asgi()

        async def _wrapper(scope, receive, send):
            if scope["type"] == "http" and scope["path"] == "/":
                collected = []
                status_code = [200]
                headers_list = [[]]

                async def _capture_send(message):
                    if message["type"] == "http.response.start":
                        status_code[0] = message["status"]
                        headers_list[0] = list(message.get("headers", []))
                    elif message["type"] == "http.response.body":
                        body = message.get("body", b"")
                        ct = dict(headers_list[0]).get(b"content-type", b"")
                        if b"text/html" in ct and b"</body>" in body:
                            body = body.replace(b"</body>", _script_tag)
                            # Update content-length
                            new_headers = [
                                (k, v) for k, v in headers_list[0]
                                if k != b"content-length"
                            ]
                            new_headers.append((b"content-length", str(len(body)).encode()))
                            await send({"type": "http.response.start", "status": status_code[0], "headers": new_headers})
                            await send({"type": "http.response.body", "body": body})
                            return
                        # Non-HTML, pass through
                        await send({"type": "http.response.start", "status": status_code[0], "headers": headers_list[0]})
                        await send(message)

                await stack(scope, receive, _capture_send)
            else:
                await stack(scope, receive, send)

        return _wrapper

    server_app.build_middleware_stack = _patched_build
    # Force rebuild
    server_app.middleware_stack = server_app.build_middleware_stack()


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
            # Auto-redirect to login page after successful signup
            return HTMLResponse(
                "<html><body><script>window.location.href='/welcome';</script></body></html>"
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
        inbrowser=False,
        auth=auth_fn,
        auth_message=get_login_message(),
        prevent_thread_lock=True,
        allowed_paths=[OUTPUT_DIR],
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
        _register_login_and_signup_routes(server_app)

    # Write share URL to file for easy retrieval
    share_url = app.share_url
    if share_url:
        url_file = os.path.join(os.path.dirname(__file__), "share_url.txt")
        with open(url_file, "w") as f:
            f.write(share_url)
        logging.info("Public URL: %s", share_url)
        logging.info("Student login: %s/welcome", share_url)
        print(f"\n{'='*50}")
        print(f"PUBLIC URL: {share_url}")
        print(f"STUDENT LOGIN: {share_url}/welcome")
        print(f"{'='*50}\n", flush=True)

    import webbrowser
    if is_firebase_auth_enabled():
        webbrowser.open("http://127.0.0.1:7860/welcome")
    else:
        webbrowser.open("http://127.0.0.1:7860")

    app.block_thread()
