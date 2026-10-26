"""CNN Vision on Streamlit Community Cloud: https://share.streamlit.io

Deploy with this file as the entry point. Streamlit Cloud then installs deploy/requirements.txt
(only what the site needs, CPU PyTorch) instead of the full requirements.txt with TensorFlow and CUDA.
The REST API can't run there, so the API page points to the ngrok site.
"""

import os
import runpy
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("VISION_LOCAL_URL", "https://harvest-semantic-hankering.ngrok-free.dev")

runpy.run_path(os.path.join(PROJECT_DIR, "streamlit_app.py"))
