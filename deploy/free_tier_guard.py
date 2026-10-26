"""Keeps CNN Vision on Google Cloud Run inside the free tier.

Runs only on Cloud Run (which sets K_SERVICE), started by main.py. Every 15 minutes it adds up this month's
billable instance time from Cloud Monitoring. At 80% of the free tier it removes public access to the service,
so no new instances can start, and stops this one. A Cloud Scheduler job gives public access back on the 1st
of every month, when the free tier starts again (see README).
"""

import datetime
import json
import os
import time
import urllib.parse
import urllib.request

# Free tier per month: 180,000 vCPU-seconds and 360,000 GiB-seconds.
# With 1 vCPU and 2 GiB, both run out after 180,000 seconds (50 hours) of instance time.
FREE_SECONDS = 180_000
STOP_AT = 0.8
CHECK_EVERY = 15 * 60
METADATA = "http://metadata.google.internal/computeMetadata/v1/"


def _metadata(path):
    request = urllib.request.Request(METADATA + path, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read().decode()


def _call(url, token, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {token}",
                                                              "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def used_seconds(project, service, token):
    # Billable instance time of the service since the start of this month (UTC)
    now = datetime.datetime.now(datetime.timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    query = urllib.parse.urlencode({
        "filter": 'metric.type="run.googleapis.com/container/billable_instance_time" '
                  f'AND resource.labels.service_name="{service}"',
        "interval.startTime": month_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval.endTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregation.alignmentPeriod": "3600s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
    })
    result = _call(f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{query}", token)
    return sum(float(next(iter(point["value"].values())))
               for series in result.get("timeSeries", []) for point in series["points"])


def watch(server):
    # Background thread: stop the site and API for the rest of the month before the free tier runs out
    service = os.environ["K_SERVICE"]
    project = _metadata("project/project-id")
    region = _metadata("instance/region").rsplit("/", 1)[-1]
    while not server.should_exit:
        try:
            token = json.loads(_metadata("instance/service-accounts/default/token"))["access_token"]
            used = used_seconds(project, service, token)
            print(f"[free tier] {used / 3600:.1f} of {FREE_SECONDS / 3600:.0f} free hours used this month")
            if used >= STOP_AT * FREE_SECONDS:
                _call(f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services/{service}"
                      ":setIamPolicy", token, {"policy": {"bindings": []}})
                print(f"[free tier] {STOP_AT:.0%} used - public access removed until the 1st of next month")
                server.should_exit = True
                return
        except Exception as e:  # a failed check must not take the site down
            print(f"[free tier] could not check the usage: {e}")
        time.sleep(CHECK_EVERY)
