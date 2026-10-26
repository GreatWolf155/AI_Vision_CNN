"""Shared code for the Streamlit site and the FastAPI REST API.

Keras runs on PyTorch here (the backend the models were trained with). The cloud server sets
KERAS_BACKEND=jax instead, which needs far less memory and gives the same predictions.
The backend has to be chosen before Keras is imported anywhere.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
