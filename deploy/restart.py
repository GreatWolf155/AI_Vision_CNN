"""Restart CNN Vision on Google Cloud Run after it was closed (by the charge cut-off or the free tier guard).

Checks this month's free tier usage, opens the site to the public again and resumes the monthly reopen job.
Uses your own Google Cloud login (gcloud). Run it from the "Restart CNN Vision" shortcut on the desktop, or:
    python deploy/restart.py
"""

import datetime
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

PROJECT = "cnn-vision-512817"
REGION = "us-central1"
SERVICE = "cnn-vision"
REOPEN_JOB = f"projects/{PROJECT}/locations/{REGION}/jobs/cnn-vision-reopen"
URL = "https://cnn-vision-134174130288.us-central1.run.app"
GCLOUD = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.free_tier_guard import FREE_BYTES, FREE_SECONDS, STOP_AT, month_total  # noqa: E402


def message(text, error=False):
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        (messagebox.showerror if error else messagebox.showinfo)("Restart CNN Vision", text, parent=root)
        root.destroy()
    except Exception:
        print(text)


def access_token():
    token = subprocess.run([GCLOUD, "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()
    if not token:  # not logged in (any more): the login opens in the browser
        subprocess.run([GCLOUD, "auth", "login"])
        token = subprocess.run([GCLOUD, "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()
    return token


def call(url, token, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read() or b"{}")


def site_answers():
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=90) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main():
    print("Checking this month's usage...")
    token = access_token()
    seconds = month_total(PROJECT, SERVICE, token, "billable_instance_time")
    sent = month_total(PROJECT, SERVICE, token, "network/sent_bytes_count", 'AND metric.labels.kind!="private"')
    usage = (f"This month ({datetime.date.today():%B}): {seconds / 3600:.1f} of {FREE_SECONDS / 3600:.0f} free hours, "
             f"{sent / 1e6:.0f} of {FREE_BYTES / 1e6:.0f} free MB sent.")
    if seconds >= STOP_AT * FREE_SECONDS or sent >= STOP_AT * FREE_BYTES:
        message(f"{usage}\n\nThe free tier guard would close the site again right away, so it stays closed "
                "until the 1st of next month (it reopens by itself then).")
        return 1

    print("Opening the site to the public...")
    call(f"https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}/services/{SERVICE}:setIamPolicy",
         token, {"policy": {"bindings": [{"role": "roles/run.invoker", "members": ["allUsers"]}]}})
    print("Resuming the monthly reopen job...")
    job = call(f"https://cloudscheduler.googleapis.com/v1/{REOPEN_JOB}", token, method="GET")
    if job.get("state") == "PAUSED":
        call(f"https://cloudscheduler.googleapis.com/v1/{REOPEN_JOB}:resume", token, {})

    print("Waiting for the site to start (public access takes about a minute to apply)...")
    for _attempt in range(8):
        if site_answers():
            message(f"CNN Vision is back online:\n{URL}\n\n{usage}")
            webbrowser.open(URL)
            return 0
        time.sleep(20)
    message(f"The site was reopened but did not answer yet. Try {URL} in a few minutes.\n\n{usage}", error=True)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        message(f"Could not restart CNN Vision:\n{e}", error=True)
        sys.exit(1)
