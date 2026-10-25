"""FastAPI REST API for the three CNN models.

Everything under /api is the REST API (interactive docs at /api/docs).
Every other path is passed through to the Streamlit site, so a single ngrok URL serves both.
"""

import asyncio
import os
import threading
from contextlib import asynccontextmanager

import httpx
import websockets
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Path, Query, Request, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from vision import charts, config, inference, stats

STREAMLIT_URL = os.environ.get("VISION_STREAMLIT_URL", "http://127.0.0.1:8501")
PUBLIC_URL = os.environ.get("VISION_PUBLIC_URL", "")

_models = {}
_models_lock = threading.Lock()


def get_model(model_id):
    if model_id not in config.MODELS:
        raise HTTPException(404, f"Unknown model '{model_id}'. Choose one of: {', '.join(config.MODELS)}")
    spec = config.MODELS[model_id]
    if not os.path.exists(spec.model_path):
        raise HTTPException(503, f"The model file is missing: {spec.model_path}")
    with _models_lock:
        if model_id not in _models:
            _models[model_id] = inference.load_model(spec)
    return _models[model_id]


def get_stats(model_id):
    lm = get_model(model_id)
    try:
        return stats.get_stats(lm)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Statistics need the test images: {e}") from e


def _warm_up():
    # Load the models and compute any missing statistics in the background, so the first request is fast
    for spec in inference.available_models():
        try:
            stats.get_stats(get_model(spec.id))
        except Exception as e:  # a missing dataset shouldn't stop the API
            print(f"[api] could not prepare {spec.name}: {e}")
    print("[api] models loaded and statistics ready")


@asynccontextmanager
async def lifespan(app):
    app.state.http = httpx.AsyncClient(base_url=STREAMLIT_URL, timeout=None)
    threading.Thread(target=_warm_up, daemon=True).start()
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="CNN Vision API",
    description=(
        "REST API for three CNN image classifiers trained in `reference work.ipynb`: "
        "**fruits & vegetables**, **face masks** and **facial expressions**.\n\n"
        "Predictions below the confidence threshold (60% by default) are returned as `Unknown`.\n\n"
        "When calling the API through a free ngrok URL from code, add the header "
        "`ngrok-skip-browser-warning: 1` to skip ngrok's browser warning page."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
    swagger_ui_parameters={"syntaxHighlight.theme": "monokai", "defaultModelsExpandDepth": 0},
)


ModelId = Annotated[str, Path(description="Which model: `fruits`, `face-mask` or `emotions`", examples=["fruits"])]
GraphName = Annotated[str, Path(description="`per_class_metrics`, `confusion_matrix` or `class_distribution`")]


# ---------- Response schemas ----------

class ModelInfo(BaseModel):
    id: str = Field(examples=["fruits"])
    name: str
    title: str
    icon: str
    description: str
    classes: list[str]
    labels: dict[str, str]
    image_size: list[int]
    color_mode: str
    face_crop: bool
    endpoints: dict[str, str]


class ClassProbability(BaseModel):
    model_config = {"populate_by_name": True}
    class_: str = Field(alias="class")
    label: str
    probability: float


class PredictionResponse(BaseModel):
    model_config = {"populate_by_name": True}
    model_id: str
    label: str = Field(description="The predicted label, or 'Unknown' if the confidence is below the threshold")
    class_: str = Field(alias="class", description="The most likely class, even when the label is 'Unknown'")
    class_label: str
    confidence: float
    is_unknown: bool
    threshold: float
    face_found: bool | None
    probabilities: list[ClassProbability]
    notes: list[str]


def _model_info(spec, lm):
    base = f"/api/models/{spec.id}"
    return ModelInfo(
        id=spec.id, name=spec.name, title=spec.title, icon=spec.icon, description=spec.tagline,
        classes=lm.classes, labels={cls: spec.label(cls) for cls in lm.classes},
        image_size=list(lm.image_size), color_mode=lm.color_mode, face_crop=spec.face_crop,
        endpoints={"predict": f"{base}/predict", "stats": f"{base}/stats", "report": f"{base}/stats/report"},
    )


def _graph_urls(model_id):
    return {name: f"/api/models/{model_id}/stats/graphs/{name}.png" for name in charts.PNG_GRAPHS}


# ---------- API routes ----------

@app.get("/api", tags=["General"], summary="API overview")
def api_root():
    """Links to the docs, the models and the statistics."""
    return {
        "name": "CNN Vision API",
        "docs": "/api/docs",
        "public_url": PUBLIC_URL or None,
        "models": [f"/api/models/{spec.id}" for spec in inference.available_models()],
        "statistics": "/api/stats",
    }


@app.get("/api/health", tags=["General"], summary="Health check")
def health():
    """Whether the API is running, and which models are available and loaded."""
    return {
        "status": "ok",
        "models_available": [spec.id for spec in inference.available_models()],
        "models_loaded": sorted(_models),
    }


@app.get("/api/models", tags=["Models"], summary="List the models", response_model=list[ModelInfo])
def list_models():
    """Every trained model with its classes, friendly labels, input size and color mode."""
    return [_model_info(spec, get_model(spec.id)) for spec in inference.available_models()]


@app.get("/api/models/{model_id}", tags=["Models"], summary="Describe one model", response_model=ModelInfo)
def model_details(model_id: ModelId):
    """The classes, friendly labels, input size and color mode of one model."""
    lm = get_model(model_id)
    return _model_info(lm.spec, lm)


@app.post(
    "/api/models/{model_id}/predict", tags=["Prediction"], summary="Classify an image",
    response_model=PredictionResponse, response_model_by_alias=True,
)
async def predict(
    model_id: ModelId,
    file: UploadFile = File(..., description="A JPG, PNG or WEBP image"),
    threshold: float = Query(config.DEFAULT_THRESHOLD, ge=0, le=1, description="Below this confidence the label is 'Unknown'"),
    face_crop: bool | None = Query(None, description="Crop to the face first (default: on for facial expressions)"),
):
    """Upload an image and get the predicted class with the probability of every option.
    Below the threshold the label is `Unknown`."""
    lm = get_model(model_id)
    data = await file.read()
    try:
        img = inference.open_image(data)
    except Exception as e:
        raise HTTPException(400, "The file is not an image that can be opened") from e
    result = await asyncio.to_thread(inference.predict, lm, img, threshold, face_crop)
    return {"model_id": model_id, **result.as_dict(lm.spec)}


@app.get("/api/stats", tags=["Statistics"], summary="Accuracy summary of all models")
def all_stats():
    """Test accuracy and F1-scores of every model, with links to their full statistics and graphs."""
    summary = []
    for spec in inference.available_models():
        s = get_stats(spec.id)
        summary.append({
            "model_id": spec.id,
            "model_name": spec.name,
            "accuracy": s["accuracy"],
            "macro_f1": s["macro_avg"]["f1"],
            "weighted_f1": s["weighted_avg"]["f1"],
            "test_images": s["test_images"],
            "classes": len(s["classes"]),
            "stats": f"/api/models/{spec.id}/stats",
            "report": f"/api/models/{spec.id}/stats/report",
            "graphs": _graph_urls(spec.id),
        })
    return {"models": summary, "graphs": {"accuracy_overview": "/api/stats/graphs/accuracy_overview.png"}}


@app.get("/api/stats/graphs/accuracy_overview.png", tags=["Statistics"], summary="Graph: accuracy of all models",
         response_class=Response, responses={200: {"content": {"image/png": {}}}})
def accuracy_overview_graph():
    """A PNG bar chart comparing the test accuracy of all models."""
    all_s = [get_stats(spec.id) for spec in inference.available_models()]
    return Response(png_cached("overview", tuple(s["computed_at"] for s in all_s), lambda: charts.png_accuracy_overview(all_s)),
                    media_type="image/png")


@app.get("/api/models/{model_id}/stats", tags=["Statistics"], summary="Full statistics of one model")
def model_stats(model_id: ModelId):
    """Accuracy, per-class precision / recall / F1, confusion matrix, confidence, dataset sizes and architecture,
    measured on the model's test images."""
    s = get_stats(model_id)
    result = {k: v for k, v in s.items() if k != "fingerprint"}
    result["graphs"] = _graph_urls(model_id)
    result["report"] = f"/api/models/{model_id}/stats/report"
    return result


_png_cache = {}
_png_lock = threading.Lock()  # matplotlib can only draw one figure at a time


def png_cached(key, version, render):
    # Keyed on the stats' timestamp, so graphs are redrawn only when the statistics change
    with _png_lock:
        if _png_cache.get(key, (None,))[0] != version:
            _png_cache[key] = (version, render())
        return _png_cache[key][1]


@app.get("/api/models/{model_id}/stats/graphs/{graph}.png", tags=["Statistics"], summary="Graph of one model",
         response_class=Response, responses={200: {"content": {"image/png": {}}}})
def model_graph(model_id: ModelId, graph: GraphName):
    """A PNG graph of one model: `per_class_metrics`, `confusion_matrix` or `class_distribution`."""
    if graph not in charts.PNG_GRAPHS:
        raise HTTPException(404, f"Unknown graph '{graph}'. Choose one of: {', '.join(charts.PNG_GRAPHS)}")
    s = get_stats(model_id)
    render = charts.PNG_GRAPHS[graph]
    return Response(png_cached(f"{model_id}/{graph}", s["computed_at"], lambda: render(s)), media_type="image/png")


@app.get("/api/models/{model_id}/stats/report", tags=["Statistics"], summary="Statistics as a web page with graphs",
         response_class=HTMLResponse)
def model_report(model_id: ModelId):
    """The statistics of one model as a web page, with a metrics table and all the graphs."""
    s = get_stats(model_id)
    spec = config.MODELS[model_id]
    rows = "".join(
        f"<tr><td>{r['label']}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
        f"<td>{r['f1']:.3f}</td><td>{r['support']:,}</td></tr>"
        for r in s["per_class"]
    )
    graphs = "".join(f'<img src="{url}" alt="{name.replace("_", " ")}">' for name, url in _graph_urls(model_id).items())
    tiles = [
        ("Test accuracy", f"{s['accuracy']:.3f}"),
        ("Macro F1", f"{s['macro_avg']['f1']:.3f}"),
        ("Test images", f"{s['test_images']:,}"),
        ("Classes", str(len(s["classes"]))),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in tiles)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{spec.title} - statistics</title>
<style>
body{{margin:0;font-family:'Segoe UI',system-ui,sans-serif;background:#f5f4fd;color:{config.INK}}}
header{{background:linear-gradient(120deg,{config.BLUE},{config.PURPLE});color:#fff;padding:28px 16px}}
header h1{{margin:0;font-size:1.6rem}} header p{{margin:6px 0 0;opacity:.85}}
main{{max-width:980px;margin:0 auto;padding:16px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:8px 0 20px}}
.tile{{background:#fff;border-radius:14px;padding:16px;box-shadow:0 1px 3px #1e1b4b14}}
.v{{font-size:1.7rem;font-weight:700;color:{config.INDIGO}}} .k{{color:{config.INK_MUTED};font-size:.9rem}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 1px 3px #1e1b4b14}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid {config.GRID}}} th{{background:#eeecfb}}
img{{width:100%;background:#fff;border-radius:14px;margin:16px 0 0;box-shadow:0 1px 3px #1e1b4b14}}
a{{color:{config.INDIGO}}}
</style></head><body><header><h1>{spec.icon} {spec.title}</h1><p>Model statistics on {s['test_images']:,} test images
- computed {s['computed_at']}</p></header><main><div class="tiles">{tiles_html}</div>
<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test images</th></tr>{rows}</table>
{graphs}<p><a href="/api/models/{model_id}/stats">Raw JSON</a> · <a href="/api/docs">API docs</a> · <a href="/">Back to the site</a></p>
</main></body></html>"""


# ---------- Pass everything else to Streamlit ----------

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}


@app.websocket("/{path:path}")
async def proxy_websocket(ws: WebSocket, path: str):
    # Streamlit talks to the browser over a WebSocket (/_stcore/stream)
    query = ws.url.query
    target = STREAMLIT_URL.replace("http", "ws", 1) + "/" + path + (f"?{query}" if query else "")
    subprotocols = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    # Pass on the headers Streamlit reads (the user agent is how the site spots phones)
    headers = {k: v for k, v in ws.headers.items()
               if k.lower() in ("user-agent", "cookie", "accept-language", "x-forwarded-for")}
    try:
        upstream = await websockets.connect(
            target, subprotocols=subprotocols or None, additional_headers=headers,
            max_size=None, ping_interval=None, compression=None,
        )
    except Exception:
        await ws.close(code=1011)
        return

    await ws.accept(subprotocol=upstream.subprotocol)

    async def browser_to_streamlit():
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await upstream.send(message["bytes"])
                elif message.get("text") is not None:
                    await upstream.send(message["text"])
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            pass

    async def streamlit_to_browser():
        try:
            async for message in upstream:
                if isinstance(message, bytes):
                    await ws.send_bytes(message)
                else:
                    await ws.send_text(message)
        except (WebSocketDisconnect, websockets.ConnectionClosed, RuntimeError):
            pass

    tasks = [asyncio.create_task(browser_to_streamlit()), asyncio.create_task(streamlit_to_browser())]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in tasks:
        task.cancel()
    await upstream.close()
    try:
        await ws.close()
    except (RuntimeError, WebSocketDisconnect):
        pass  # the browser already left


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
               include_in_schema=False)
async def proxy_http(request: Request, path: str):
    client: httpx.AsyncClient = request.app.state.http
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    upstream_request = client.build_request(
        request.method, "/" + path, params=request.query_params, headers=headers, content=await request.body(),
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.ConnectError:
        return HTMLResponse("<h2>The site is starting up...</h2><p>Refresh in a few seconds.</p>", status_code=503)
    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
    return StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code, headers=response_headers,
        background=_close_later(upstream),
    )


def _close_later(upstream):
    from starlette.background import BackgroundTask

    return BackgroundTask(upstream.aclose)
