"""CNN Vision - start the whole project with one command:

    python main.py

This starts:
  * the Streamlit site (classify photos, model statistics) - on desktop and phone
  * the FastAPI REST API (under /api, docs at /api/docs)
  * an ngrok tunnel, so the site can be opened from anywhere, e.g. your phone

Both the site and the API are served from one address (the API passes the site through),
so a single ngrok URL is enough.

The site runs from the GitHub repo: when started from a clone, it first pulls the latest
version, then checks GitHub every few minutes and restarts itself with any new commits.

Options:
    python main.py --no-ngrok      only on this computer (http://localhost:8765)
    python main.py --reset-token   ask for the ngrok authtoken again
    python main.py --domain NAME   use your free static ngrok domain, e.g. my-name.ngrok-free.app
    python main.py --no-update     don't pull new versions from GitHub
"""

import argparse
import getpass
import os
import re
import subprocess
import sys
import time
import urllib.request
import webbrowser

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PAGE = "https://dashboard.ngrok.com/get-started/your-authtoken"

# Emoji and the QR code need UTF-8 in the Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Start the CNN Vision site, REST API and ngrok tunnel.")
    parser.add_argument("--port", type=int, default=8765, help="port of the site + API (default 8765)")
    parser.add_argument("--streamlit-port", type=int, default=8501, help="internal Streamlit port (default 8501)")
    parser.add_argument("--no-ngrok", action="store_true", help="don't open an ngrok tunnel")
    parser.add_argument("--reset-token", action="store_true", help="ask for the ngrok authtoken again")
    parser.add_argument("--domain", help="your static ngrok domain (optional)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the site in the browser")
    parser.add_argument("--no-update", action="store_true", help="don't pull new versions from GitHub")
    parser.add_argument("--update-every", type=float, default=5, metavar="MINUTES",
                        help="how often to check GitHub for a new version (default 5)")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)  # internal: run the servers
    args = parser.parse_args()
    if args.serve and "--update-every" not in sys.argv:
        args.update_every = 0  # the supervisor passes it on only when updating is on
    return args


# ---------- ngrok ----------

def saved_ngrok_token():
    # The authtoken saved in ngrok's own config file (outside the project), if there is one
    from pyngrok import conf

    path = conf.get_default().config_path or conf.DEFAULT_NGROK_CONFIG_PATH
    try:
        with open(path, encoding="utf-8") as f:
            match = re.search(r"^\s*authtoken:\s*(\S+)", f.read(), re.MULTILINE)
    except OSError:
        return None
    return match.group(1).strip("'\"") if match else None


def token_window(message=""):
    # A small window asking for the token. Returns the token, "" if cancelled, or None if no window can be shown.
    try:
        import tkinter as tk
    except ImportError:
        return None
    try:
        root = tk.Tk()
    except tk.TclError:  # no screen to show it on
        return None

    result = {"token": ""}
    root.title("CNN Vision - ngrok authtoken")
    root.configure(bg="#f7f6fe")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    header = tk.Frame(root, bg="#4c3fd1", padx=20, pady=14)
    header.pack(fill="x")
    tk.Label(header, text="🔮 Put CNN Vision online", fg="white", bg="#4c3fd1",
             font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(header, text="ngrok needs your authtoken. It's saved in ngrok's own settings, not in the project.",
             fg="#e6e4ff", bg="#4c3fd1", font=("Segoe UI", 9)).pack(anchor="w")

    body = tk.Frame(root, bg="#f7f6fe", padx=20, pady=14)
    body.pack(fill="both")
    if message:
        tk.Label(body, text=message, fg="#c0392b", bg="#f7f6fe", font=("Segoe UI", 9, "bold")).pack(anchor="w")
    tk.Label(body, text="Authtoken:", bg="#f7f6fe", fg="#1e1b4b", font=("Segoe UI", 10)).pack(anchor="w")
    entry = tk.Entry(body, show="•", width=52, font=("Consolas", 11), relief="solid", bd=1)
    entry.pack(fill="x", pady=(2, 6), ipady=4)
    show = tk.BooleanVar(value=False)
    tk.Checkbutton(body, text="Show token", variable=show, bg="#f7f6fe", activebackground="#f7f6fe",
                   command=lambda: entry.config(show="" if show.get() else "•")).pack(anchor="w")
    link = tk.Label(body, text="Get your token: dashboard.ngrok.com → Your Authtoken", fg="#3b5bdb",
                    bg="#f7f6fe", cursor="hand2", font=("Segoe UI", 9, "underline"))
    link.pack(anchor="w", pady=(6, 0))
    link.bind("<Button-1>", lambda _e: webbrowser.open(TOKEN_PAGE))

    def submit(_event=None):
        result["token"] = entry.get().strip()
        if result["token"]:
            root.destroy()

    buttons = tk.Frame(body, bg="#f7f6fe")
    buttons.pack(fill="x", pady=(14, 0))
    tk.Button(buttons, text="Start online", command=submit, bg="#5b47e0", fg="white", activebackground="#7c3aed",
              activeforeground="white", relief="flat", padx=16, pady=6, font=("Segoe UI", 10, "bold")).pack(side="right")
    tk.Button(buttons, text="Skip (local only)", command=root.destroy, relief="flat", padx=12, pady=6,
              bg="#eceafb", font=("Segoe UI", 10)).pack(side="right", padx=8)

    root.bind("<Return>", submit)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 3
    root.geometry(f"+{x}+{y}")
    root.focus_force()
    entry.focus_set()
    root.mainloop()
    return result["token"]


def ask_for_token(message=""):
    # Returns the token, or None to continue without ngrok
    print("\nngrok needs an authtoken to put the site online - a window is asking for it.")
    token = token_window(message)
    if token is not None:
        return token or None
    # No window possible (e.g. over SSH): ask in the terminal instead
    print(f"Copy your authtoken from: {TOKEN_PAGE}")
    token = getpass.getpass("Paste it (hidden) and press Enter, or just Enter to skip: ").strip()
    return token or None


def start_ngrok(port, domain=None, reset_token=False):
    # Returns the public https address, or None if ngrok could not start
    from pyngrok import ngrok
    from pyngrok.exception import PyngrokError

    token = os.environ.get("NGROK_AUTHTOKEN")
    if reset_token or not (token or saved_ngrok_token()):
        token = ask_for_token()
        if token is None:
            print("[ngrok] skipped - running on this computer only.")
            return None
    for attempt in range(6):
        try:
            if token:
                ngrok.set_auth_token(token)  # saved in ngrok's config, so you're only asked once
            options = {"domain": domain} if domain else {}
            tunnel = ngrok.connect(port, "http", **options)
            return tunnel.public_url.replace("http://", "https://")
        except PyngrokError as e:
            message = str(e)
            if "ERR_NGROK_334" in message and attempt < 5:
                # The address is still held by a copy that was just stopped - ngrok frees it within seconds
                print("[ngrok] the address is still in use by the previous run, retrying...")
                ngrok.kill()
                time.sleep(5)
                continue
            print(f"\n[ngrok] could not start: {message.strip()[:400]}")
            if "authtoken" in message.lower() or "ERR_NGROK_105" in message or "ERR_NGROK_107" in message:
                print("That authtoken was not accepted.")
                ngrok.kill()
                token = ask_for_token("That token was not accepted - please check it and try again.")
                if token is None:
                    break
                continue
            break
    print("[ngrok] continuing without a public address.")
    return None


# ---------- Servers ----------

def start_streamlit(port, env):
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    log = open(os.path.join(PROJECT_DIR, "logs", "streamlit.log"), "w", encoding="utf-8")
    command = [
        sys.executable, "-m", "streamlit", "run", os.path.join(PROJECT_DIR, "streamlit_app.py"),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",   # only reachable through the API's address
        "--server.headless", "true",
        # The API passes requests through to Streamlit, so Streamlit sees a different origin than the browser
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    return subprocess.Popen(command, cwd=PROJECT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)


def free_port(port):
    # The first port from `port` up that nothing is listening on (another program may already use it)
    import socket

    for candidate in range(port, port + 50):
        taken = False
        for host in ("127.0.0.1", "0.0.0.0"):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE if os.name == "nt" else socket.SO_REUSEADDR, 1)
                try:
                    s.bind((host, candidate))
                except OSError:
                    taken = True
                    break
        if not taken:
            if candidate != port:
                print(f"Port {port} is in use, using {candidate} instead.")
            return candidate
    raise RuntimeError(f"No free port found from {port}")


def wait_for(url, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def print_banner(local_url, public_url):
    line = "=" * 64
    print(f"\n{line}\n  🔮 CNN Vision is running\n{line}")
    print(f"  On this computer : {local_url}")
    if public_url:
        print(f"  Online (ngrok)   : {public_url}")
        print(f"  Phone            : scan the QR code below, or open the address above")
    print(f"  REST API docs    : {(public_url or local_url)}/api/docs")
    print(f"  Model statistics : {(public_url or local_url)}/api/stats")
    print(line)
    if public_url:
        try:
            import segno

            segno.make(public_url, error="m").terminal(compact=True)
        except Exception:
            pass
        print("  ngrok shows a one-time warning page first - press 'Visit Site'.")
    print("  Press Ctrl+C to stop.\n")


# ---------- Keeping the site in sync with the GitHub repo ----------

RESTART_FOR_UPDATE = 75  # exit code of the server process when GitHub has new commits


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None


def tracks_github():
    # Whether this folder is a clone of the GitHub repo (a git checkout with an upstream branch)
    result = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    return result is not None and result.returncode == 0


def pull_from_github():
    result = git("pull", "--ff-only")
    if result is None or result.returncode != 0:
        print(f"[update] could not pull from GitHub, running the current files:\n{(result.stderr if result else '').strip()}")
    elif "Already up to date" not in result.stdout:
        print("[update] pulled the latest version from GitHub")


def github_has_update():
    fetched = git("fetch", "--quiet")
    if fetched is None or fetched.returncode != 0:
        return False
    head, upstream = git("rev-parse", "HEAD"), git("rev-parse", "@{u}")
    if not head or not upstream or head.stdout == upstream.stdout:
        return False
    # Only updates that can be applied cleanly (the local copy has no commits of its own)
    return git("merge-base", "--is-ancestor", "HEAD", "@{u}").returncode == 0


def watch_github(server, minutes):
    # Background thread: when GitHub has new commits, stop the server so the supervisor can pull and restart it
    while not server.should_exit:
        time.sleep(minutes * 60)
        if github_has_update():
            print("[update] GitHub has a new version - restarting with it...")
            server.restart_for_update = True
            server.should_exit = True
            return


def supervise(args):
    # Pull the latest version from GitHub, run the site, and run it again whenever GitHub has new commits
    updating = not args.no_update and tracks_github()
    if updating:
        pull_from_github()
    elif not args.no_update:
        print("[update] this folder is not a clone of the GitHub repo, so it won't update itself")

    child_args = sys.argv[1:]
    while True:
        command = [sys.executable, os.path.abspath(__file__), "--serve", *child_args]
        if updating:
            command += ["--update-every", str(args.update_every)]
        process = subprocess.Popen(command, cwd=PROJECT_DIR)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            code = process.wait()
        if code == RESTART_FOR_UPDATE and updating:
            pull_from_github()
            # The browser is already open, and the ngrok token was already entered
            child_args = [a for a in child_args if a != "--reset-token"] + ["--no-browser"]
            continue
        return code


def serve(args):
    args.port = free_port(args.port)
    args.streamlit_port = free_port(args.streamlit_port if args.streamlit_port != args.port else args.port + 1)
    local_url = f"http://localhost:{args.port}"
    streamlit_url = f"http://127.0.0.1:{args.streamlit_port}"

    public_url = None if args.no_ngrok else start_ngrok(args.port, args.domain, args.reset_token)

    env = dict(os.environ, KERAS_BACKEND="torch", VISION_STREAMLIT_URL=streamlit_url, VISION_LOCAL_URL=local_url,
               VISION_PUBLIC_URL=public_url or "", PYTHONIOENCODING="utf-8")
    os.environ.update(env)  # the API (started below in this process) reads the same settings

    print("Starting the Streamlit site...")
    streamlit = start_streamlit(args.streamlit_port, env)
    if not wait_for(f"{streamlit_url}/_stcore/health"):
        print("Streamlit did not start - see logs/streamlit.log")

    import uvicorn

    from api import app

    print_banner(local_url, public_url)
    if not args.no_browser:
        webbrowser.open(local_url)

    import threading

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="warning",
                                           ws_max_size=200 * 1024 * 1024))
    server.restart_for_update = False
    if args.update_every:
        print(f"[update] checking GitHub for a new version every {args.update_every:g} minutes")
        threading.Thread(target=watch_github, args=(server, args.update_every), daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        streamlit.terminate()
        if public_url:
            from pyngrok import ngrok

            ngrok.kill()
    return RESTART_FOR_UPDATE if server.restart_for_update else 0


def main():
    args = parse_args()
    if args.serve:
        sys.exit(serve(args))
    sys.exit(supervise(args))


if __name__ == "__main__":
    main()
