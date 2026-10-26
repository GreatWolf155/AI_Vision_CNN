"""Closes CNN Vision as soon as it causes a real charge (a Google Cloud Function).

The project's budget sends its month-to-date cost here several times a day (through Pub/Sub). The cost
already has the free trial credit taken off, so it is 0 while everything is free. When it is above 0, this
closes the site to the public (which stops all its running time and traffic) and pauses the monthly reopen
job, so the site stays closed until it is restarted by hand (deploy/restart.ps1).
"""

import base64
import json
import os
import urllib.request

import functions_framework

PROJECT = os.environ["PROJECT"]
REGION = os.environ["REGION"]
SERVICE = os.environ["SERVICE"]
REOPEN_JOB = os.environ["REOPEN_JOB"]
DRY_RUN = os.environ.get("DRY_RUN") == "1"  # log what would happen, change nothing (for testing)


def _token():
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())["access_token"]


def _post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {_token()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


@functions_framework.cloud_event
def on_budget(event):
    budget = json.loads(base64.b64decode(event.data["message"]["data"]))
    cost, currency = budget["costAmount"], budget.get("currencyCode", "")
    if cost <= 0:
        print(f"[cutoff] no charges this month ({cost} {currency})")
        return
    print(f"[cutoff] real charge of {cost} {currency} - closing the site{' (dry run)' if DRY_RUN else ''}")
    if DRY_RUN:
        return
    _post(f"https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}/services/{SERVICE}:setIamPolicy",
          {"policy": {"bindings": []}})
    _post(f"https://cloudscheduler.googleapis.com/v1/{REOPEN_JOB}:pause", {})
    print("[cutoff] site closed and the monthly reopen paused - run 'Restart CNN Vision' to open it again")
