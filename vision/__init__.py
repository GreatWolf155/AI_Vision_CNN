"""Shared code for the Streamlit site and the FastAPI REST API.

Keras must run on the same backend the models were trained with (PyTorch),
and the backend has to be chosen before Keras is imported anywhere.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
