"""Model statistics: accuracy, per-class metrics and confusion matrix on the test images.

This is the evaluation from section 6.6 of the notebook. It needs the test images, so the
results are saved in models/<name>.stats.json (which is in the GitHub repo, unlike the datasets)
and recomputed only when the .pkl changes.
"""

import functools
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .inference import list_images, predict_batch, prepare

STATS_VERSION = 2
_locks = {spec_id: threading.Lock() for spec_id in config.MODELS}


@functools.lru_cache(maxsize=16)
def _sha256(path, size, mtime):
    # size and mtime are only part of the cache key, so the file is hashed again when it changes
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_fingerprint(spec):
    # Based on the file's contents, not its date - a fresh clone of the repo gives every file a new date
    st = os.stat(spec.model_path)
    return {"sha256": _sha256(spec.model_path, st.st_size, st.st_mtime), "version": STATS_VERSION}


def can_evaluate(spec):
    # Whether the test images are here (they are not in the GitHub repo)
    return os.path.isdir(spec.test_dir)


def load_cached(spec):
    # The cached statistics, or None if there are none or the model has changed since
    try:
        with open(spec.stats_path) as f:
            stats = json.load(f)
    except (OSError, ValueError):
        return None
    if stats.get("fingerprint") != _model_fingerprint(spec):
        return None
    return stats


def count_images(folder, classes):
    counts = {c: 0 for c in classes}
    for _path, index in list_images(folder, classes):
        counts[classes[index]] += 1
    return counts


def _architecture(model):
    layers = []
    for layer in model.layers:
        try:
            shape = list(layer.output.shape[1:])
        except Exception:
            shape = None
        layers.append({
            "name": layer.name,
            "type": type(layer).__name__,
            "output_shape": shape,
            "params": int(layer.count_params()),
        })
    return layers


def _threshold_curve(confidence, correct):
    curve = []
    for threshold in np.round(np.arange(0, 1.0, 0.05), 2):
        answered = confidence >= threshold
        curve.append({
            "threshold": float(threshold),
            "answered": int(answered.sum()),
            "answered_fraction": float(answered.mean()),
            "accuracy": float(correct[answered].mean()) if answered.any() else None,
        })
    return curve


def evaluate(lm, batch_size=256, progress=None):
    # Predict every test image, then compare the predictions with the real classes.
    # progress(done, total) is called after every batch, if given.
    spec = lm.spec
    items = list_images(spec.test_dir, lm.classes)
    if not items:
        raise FileNotFoundError(f"No test images found in {spec.test_dir}")

    def load(path):
        # No face/square cropping: the test images are already prepared like the training images
        with Image.open(path) as img:
            batch, _, _ = prepare(lm, img.convert("RGB"), face_crop=False, square_crop=False)
        return batch[0]

    started = time.time()
    probabilities = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(items), batch_size):
            paths = [p for p, _ in items[start:start + batch_size]]
            batch = np.stack(list(pool.map(load, paths)))
            probabilities.append(predict_batch(lm, batch))
            if progress:
                progress(min(start + batch_size, len(items)), len(items))
    probabilities = np.concatenate(probabilities)

    actual = np.array([index for _, index in items])
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == actual

    report = classification_report(
        actual, predicted, labels=list(range(len(lm.classes))),
        target_names=lm.classes, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(lm.classes))))

    per_class = []
    for i, c in enumerate(lm.classes):
        r = report[c]
        per_class.append({
            "class": c,
            "label": spec.label(c),
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "support": int(r["support"]),
        })

    # The most common mistakes, e.g. "Eggplant predicted as Avocado: 70 times"
    confusions = []
    for i in range(len(lm.classes)):
        for j in range(len(lm.classes)):
            if i != j and matrix[i, j] > 0:
                confusions.append({
                    "actual": lm.classes[i],
                    "predicted": lm.classes[j],
                    "count": int(matrix[i, j]),
                    "share_of_actual": float(matrix[i, j] / matrix[i].sum()),
                })
    confusions.sort(key=lambda c: c["count"], reverse=True)

    # How many predictions the "Unknown" bonus rule would hide, and how accurate the rest are
    unknown = confidence < config.DEFAULT_THRESHOLD
    confident = ~unknown

    model = lm.model
    stats = {
        "fingerprint": _model_fingerprint(spec),
        "model_id": spec.id,
        "model_name": spec.name,
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_seconds": round(time.time() - started, 1),
        "classes": lm.classes,
        "labels": [spec.label(c) for c in lm.classes],
        "test_images": int(len(items)),
        "accuracy": float(correct.mean()),
        "macro_avg": {k.replace("-score", ""): float(v) for k, v in report["macro avg"].items() if k != "support"},
        "weighted_avg": {k.replace("-score", ""): float(v) for k, v in report["weighted avg"].items() if k != "support"},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": confusions[:10],
        "confidence": {
            "mean_when_correct": float(confidence[correct].mean()) if correct.any() else None,
            "mean_when_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
            "threshold": config.DEFAULT_THRESHOLD,
            "share_unknown": float(unknown.mean()),
            "accuracy_when_confident": float(correct[confident].mean()) if confident.any() else None,
            # What each "Unknown" threshold would do: how many images still get an answer, and how accurate those are
            "threshold_curve": _threshold_curve(confidence, correct),
        },
        "dataset": {
            "test_counts": {c: int((actual == i).sum()) for i, c in enumerate(lm.classes)},
            "train_counts": count_images(spec.train_dir, lm.classes),
        },
        "model": {
            "input_shape": list(model.input_shape[1:]),
            "image_size": list(lm.image_size),
            "color_mode": lm.color_mode,
            "total_params": int(model.count_params()),
            "layers": _architecture(model),
            "file_size_mb": round(os.path.getsize(spec.model_path) / 1024 ** 2, 1),
            "saved_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(spec.model_path))),
        },
    }
    return stats


def get_stats(lm, progress=None):
    # Cached statistics, computing them first if needed (only one computation per model at a time)
    spec = lm.spec
    cached = load_cached(spec)
    if cached is not None:
        return cached
    with _locks[spec.id]:
        cached = load_cached(spec)  # another thread may have just finished
        if cached is not None:
            return cached
        stats = evaluate(lm, progress=progress)
        partial = spec.stats_path + ".partial"
        with open(partial, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(partial, spec.stats_path)
        return stats
