"""Keeps CNN Vision on Google Cloud Run inside the free tier.

Runs only on Cloud Run (which sets K_SERVICE), started by main.py. Every 15 minutes it reads this month's
usage from Cloud Monitoring. At 80% of the free running time or of the free outgoing traffic it closes the
site to the public, so no new instances can start, and stops this one. A Cloud Scheduler job opens it again
on the 1st of every month, when the free tier starts again (see README).
"""

import datetime
import json
import os
import time
import urllib.parse
import urllib.request

# Free tier per month: 180,000 vCPU-seconds and 360,000 GiB-seconds - with 1 vCPU and 2 GiB both run out after
# 180,000 seconds (50 hours) of instance time - and 1 GB of outgoing traffic from a North American region
FREE_SECONDS = 180_000
FREE_BYTES = 1_000_000_000
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


def month_total(project, service, token, metric, extra_filter=""):
    # Sum of a Cloud Run metric for the service since the start of this month (UTC)
    now = datetime.datetime.now(datetime.timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    query = urllib.parse.urlencode({
        "filter": f'metric.type="run.googleapis.com/container/{metric}" '
                  f'AND resource.labels.service_name="{service}" {extra_filter}',
        "interval.startTime": month_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval.endTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregation.alignmentPeriod": "3600s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
    })
    result = _call(f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{query}", token)
    return sum(float(next(iter(point["value"].values())))
               for series in result.get("timeSeries", []) for point in series["points"])


def close_site(project, region, service, token):
    # Remove public access: visitors get "403 Forbidden" and no new instances start
    _call(f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services/{service}:setIamPolicy",
          token, {"policy": {"bindings": []}})


def watch(server):
    # Background thread: stop the site and API for the rest of the month before the free tier runs out
    service = os.environ["K_SERVICE"]
    project = _metadata("project/project-id")
    region = _metadata("instance/region").rsplit("/", 1)[-1]
    while not server.should_exit:
        try:
            token = json.loads(_metadata("instance/service-accounts/default/token"))["access_token"]
            seconds = month_total(project, service, token, "billable_instance_time")
            # Everything sent out except traffic inside Google's private network
            sent = month_total(project, service, token, "network/sent_bytes_count",
                               'AND metric.labels.kind!="private"')
            print(f"[free tier] this month: {seconds / 3600:.1f} of {FREE_SECONDS / 3600:.0f} free hours, "
                  f"{sent / 1e6:.0f} of {FREE_BYTES / 1e6:.0f} free MB sent")
            if seconds >= STOP_AT * FREE_SECONDS or sent >= STOP_AT * FREE_BYTES:
                close_site(project, region, service, token)
                print(f"[free tier] {STOP_AT:.0%} used - public access removed until the 1st of next month")
                server.should_exit = True
                return
        except Exception as e:  # a failed check must not take the site down
            print(f"[free tier] could not check the usage: {e}")
        time.sleep(CHECK_EVERY)
