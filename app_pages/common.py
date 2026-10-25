"""Shared pieces of the Streamlit pages: model loading, the model menu, phone detection and styles."""

import os
import re

import streamlit as st

from vision import config, inference, stats

PUBLIC_URL = os.environ.get("VISION_PUBLIC_URL", "")
LOCAL_URL = os.environ.get("VISION_LOCAL_URL", "http://localhost:8765")
MOBILE_AGENTS = re.compile(r"Mobi|Android|iPhone|iPad|iPod|Windows Phone|webOS|BlackBerry", re.IGNORECASE)
VIEW_MODES = ["Auto", "Phone", "Desktop"]
SHORT_NAMES = {"fruits": "Fruits", "face-mask": "Mask", "emotions": "Emotions"}  # phone-sized menu labels


# ---------- Models ----------

@st.cache_resource(show_spinner=False)
def load_model(model_id):
    # Loaded from disk once, when the site first needs it, then kept in memory for every visitor
    return inference.load_model(config.MODELS[model_id])


def get_model(model_id):
    spec = config.MODELS[model_id]
    with st.spinner(f"Loading the {spec.title} model..."):
        return load_model(model_id)


@st.cache_data(show_spinner=False)
def _cached_stats(model_id, model_mtime):
    return stats.get_stats(load_model(model_id))


def get_stats(model_id):
    # The model's statistics, or None if they were never computed and the test images aren't here to compute them
    spec = config.MODELS[model_id]
    if stats.load_cached(spec) is None:
        if not stats.can_evaluate(spec):
            return None
        with st.spinner("Evaluating the model on the test images (only needed once)..."):
            return _cached_stats(model_id, os.path.getmtime(spec.model_path))
    return _cached_stats(model_id, os.path.getmtime(spec.model_path))


def available_ids():
    return [spec.id for spec in inference.available_models()]


# ---------- Values that survive switching pages ----------
# Streamlit forgets a widget's value when its page is left, so each widget writes into a permanent key

def _keep(key):
    st.session_state[key] = st.session_state[f"_{key}"]


def bind(key, default):
    st.session_state.setdefault(key, default)
    st.session_state[f"_{key}"] = st.session_state[key]
    return {"key": f"_{key}", "on_change": _keep, "args": (key,)}


def init_state():
    ids = available_ids()
    params = st.query_params
    if "model_id" not in st.session_state:
        wanted = params.get("model", config.DEFAULT_MODEL)
        st.session_state.model_id = wanted if wanted in ids else (ids[0] if ids else config.DEFAULT_MODEL)
    if "view_mode" not in st.session_state:
        wanted = params.get("view", "auto").capitalize()
        st.session_state.view_mode = wanted if wanted in VIEW_MODES else "Auto"
    st.session_state.setdefault("threshold", int(config.DEFAULT_THRESHOLD * 100))


def is_mobile():
    mode = st.session_state.get("view_mode", "Auto")
    if mode != "Auto":
        return mode == "Phone"
    user_agent = st.context.headers.get("User-Agent", "") or ""
    return bool(MOBILE_AGENTS.search(user_agent))


def current_spec():
    return config.MODELS[st.session_state.model_id]


# ---------- Widgets ----------

def model_menu(label="Choose a model"):
    # The menu for picking one of the three models
    ids = available_ids()
    if not ids:
        st.error("No trained models were found in the `models` folder. Run the notebook first.")
        st.stop()
    mobile = is_mobile()

    def fmt(model_id):
        spec = config.MODELS[model_id]
        if mobile:
            return f"{spec.icon} {SHORT_NAMES.get(model_id, spec.title)}"
        return f"{spec.icon} {spec.title}"

    st.segmented_control(
        label, ids, format_func=fmt, required=True, width="stretch",
        help="Each model was trained on a different dataset in the notebook.",
        **bind("model_id", st.session_state.model_id),
    )
    spec = current_spec()
    # Keep the choice in the address, so a reload or a shared link opens the same model
    if st.query_params.get("model") != spec.id:
        st.query_params["model"] = spec.id
    st.caption(f"{spec.tagline}. {spec.hint}")
    return spec


def threshold_slider():
    st.slider(
        "Confidence needed (below it the answer is **Unknown**)", min_value=0, max_value=100, step=5,
        format="%d%%", **bind("threshold", st.session_state.threshold),
        help="Bonus rule from the project: if the model is less sure than this, show 'Unknown'.",
    )
    return st.session_state.threshold / 100


def face_crop_toggle(spec):
    key = f"face_crop_{spec.id}"
    st.toggle(
        "Find and crop the face first", **bind(key, spec.face_crop),
        help="The training photos are close-ups of faces. Cropping to the face makes full photos look like them.",
    )
    return st.session_state[key]


def view_mode_picker():
    st.segmented_control(
        "Layout", VIEW_MODES, required=True, width="stretch",
        help="Auto picks the phone layout on phones and tablets.",
        **bind("view_mode", st.session_state.view_mode),
    )


def sidebar_footer():
    # QR code of the public ngrok address, for opening the site on a phone (desktop sidebar only)
    if PUBLIC_URL and not is_mobile():
        with st.sidebar:
            st.divider()
            st.markdown("### 📱 Open on your phone")
            st.image(qr_png(PUBLIC_URL), width=170)
            st.caption(PUBLIC_URL)


def hero(title, subtitle, icon=""):
    st.markdown(
        f'<div class="hero"><div class="hero-title">{icon} {title}</div><div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def qr_png(url):
    import io

    import segno

    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=6, border=2, dark=config.INK, light="#ffffff")
    return buf.getvalue()


# ---------- Styles ----------

def inject_css(mobile):
    st.markdown(
        f"""<style>
:root {{ --blue: {config.BLUE}; --indigo: {config.INDIGO}; --purple: {config.PURPLE}; --ink: {config.INK};
        --muted: {config.INK_MUTED}; --grid: {config.GRID}; }}
.block-container {{ padding-top: {"3.2rem" if mobile else "3rem"}; max-width: {"520px" if mobile else "1180px"}; }}
.hero {{ background: linear-gradient(120deg, var(--blue) 0%, var(--indigo) 50%, var(--purple) 100%);
         color: #fff; border-radius: 20px; padding: {"18px 18px" if mobile else "26px 30px"}; margin-bottom: 18px;
         box-shadow: 0 10px 30px -12px #4c3fd180; }}
.hero-title {{ font-size: {"1.45rem" if mobile else "2.1rem"}; font-weight: 750; line-height: 1.2; }}
.hero-sub {{ opacity: .88; margin-top: 6px; font-size: {"0.92rem" if mobile else "1.02rem"}; }}
.result-card {{ border-radius: 18px; padding: 20px 22px; margin: 4px 0 14px; color: #fff;
                background: linear-gradient(120deg, var(--blue), var(--purple)); box-shadow: 0 10px 30px -14px #4c3fd1aa; }}
.result-card.unknown {{ background: linear-gradient(120deg, #6b6894, #8f8bb8); }}
.result-kicker {{ text-transform: uppercase; letter-spacing: .08em; font-size: .75rem; opacity: .85; }}
.result-label {{ font-size: {"1.9rem" if mobile else "2.3rem"}; font-weight: 800; margin: 2px 0 8px; line-height: 1.15; }}
.result-meter {{ height: 8px; background: #ffffff40; border-radius: 99px; overflow: hidden; }}
.result-meter > div {{ height: 100%; background: #fff; border-radius: 99px; }}
.result-meta {{ margin-top: 8px; font-size: .95rem; opacity: .92; }}
.empty-state {{ border: 2px dashed #c9c4f0; border-radius: 18px; padding: 28px 18px; text-align: center;
                color: var(--muted); background: #ffffffaa; }}
.empty-state .big {{ font-size: 2.4rem; }}
.step {{ font-weight: 700; color: var(--indigo); font-size: 1.05rem; margin: 6px 0 4px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #eef0ff 0%, #f1ebff 100%); }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
    background: linear-gradient(120deg, var(--blue), var(--purple)); border: 0; }}
.stButton > button, .stDownloadButton > button {{ min-height: {"3.1rem" if mobile else "2.6rem"}; font-weight: 650; }}
[data-testid="stImage"] img {{ border-radius: 14px; {"max-height: 42vh; object-fit: contain;" if mobile else ""} }}
.tiles {{ display: grid; grid-template-columns: repeat({2 if mobile else 4}, 1fr); gap: 12px; margin: 4px 0 16px; }}
.tile {{ background: #fff; border-radius: 16px; padding: {"12px 14px" if mobile else "14px 18px"};
         box-shadow: 0 1px 3px #1e1b4b14; border: 1px solid var(--grid); }}
.tile-label {{ font-size: .88rem; color: var(--ink); font-weight: 600; }}
.tile-value {{ font-size: {"1.6rem" if mobile else "2.1rem"}; font-weight: 780; color: var(--indigo); line-height: 1.25; }}
.tile-help {{ font-size: .78rem; color: var(--muted); line-height: 1.3; }}
.chip {{ display: inline-block; background: #eceafb; color: var(--indigo); border-radius: 99px; padding: 2px 10px;
         font-size: .85rem; font-weight: 600; margin-right: 6px; }}
</style>""",
        unsafe_allow_html=True,
    )
