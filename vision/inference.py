"""Load the saved models and classify images, preparing them the same way as in training."""

import io
import json
import os
import pickle
import threading
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from . import config  # noqa: F401  (sets KERAS_BACKEND before keras is imported)

# Only one prediction runs on the GPU at a time
_predict_lock = threading.Lock()
_face_detector = None


@dataclass
class LoadedModel:
    spec: config.ModelSpec
    model: object
    classes: list
    image_size: tuple   # (height, width)
    color_mode: str     # "rgb" or "grayscale"

    @property
    def channels(self):
        return 1 if self.color_mode == "grayscale" else 3


@dataclass
class Prediction:
    class_name: str          # the most likely class
    label: str               # friendly label, or "Unknown" below the threshold
    confidence: float
    is_unknown: bool
    threshold: float
    probabilities: list      # [(class_name, probability)], most likely first
    model_input: Image.Image  # exactly what the model saw (after resizing)
    face_found: bool | None = None  # None when face cropping was not used
    notes: list = field(default_factory=list)

    def as_dict(self, spec):
        return {
            "label": self.label,
            "class": self.class_name,
            "class_label": spec.label(self.class_name),
            "confidence": round(self.confidence, 6),
            "is_unknown": self.is_unknown,
            "threshold": self.threshold,
            "face_found": self.face_found,
            "probabilities": [
                {"class": c, "label": spec.label(c), "probability": round(p, 6)}
                for c, p in self.probabilities
            ],
            "notes": self.notes,
        }


class _PredictionOnlyUnpickler(pickle.Unpickler):
    # A pickled Keras model is a call to Model._unpickle_model(<the model as a .keras file>), which also
    # rebuilds the optimizer. The notebook's optimizer is PyTorch's own Adam, which other backends can't load,
    # so load the model without it (prediction doesn't need it). Same file, same weights.
    def find_class(self, module, name):
        if (module, name) == ("builtins", "getattr"):
            return lambda obj, attr: _load_keras_file if attr == "_unpickle_model" else getattr(obj, attr)
        return super().find_class(module, name)


def _load_keras_file(fileobj):
    from keras.src.saving import saving_lib

    return saving_lib._load_model_from_fileobj(fileobj, custom_objects=None, compile=False, safe_mode=False)


def load_model(spec):
    # The notebook saved the whole Keras model with pickle, and the image settings in a .json next to it
    import keras  # noqa: F401  (imported here so the backend is set up before unpickling)

    with open(spec.model_path, "rb") as f:
        model = _PredictionOnlyUnpickler(f).load()
    with open(spec.info_path) as f:
        info = json.load(f)
    return LoadedModel(
        spec=spec,
        model=model,
        classes=info["classes"],
        image_size=tuple(info["image_size"]),
        color_mode=info["color_mode"],
    )


def available_models():
    # The models whose .pkl and .json files exist
    return [spec for spec in config.MODELS.values() if os.path.exists(spec.model_path) and os.path.exists(spec.info_path)]


def open_image(data):
    # bytes or a file-like object -> upright PIL image (phone photos store their rotation in EXIF)
    if isinstance(data, (bytes, bytearray)):
        data = io.BytesIO(data)
    img = Image.open(data)
    img = ImageOps.exif_transpose(img)
    # Transparent pixels become white, like the plain backgrounds in the training photos
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        import cv2

        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _face_detector = cv2.CascadeClassifier(path)
    return _face_detector


def crop_face(img, margin=0.15):
    # Returns (cropped image, True) for the largest face found, or (img, False) if there is none
    import cv2

    gray = np.array(img.convert("L"))
    # Detect on a smaller copy - much faster on large phone photos
    scale = min(1.0, 640 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    small = cv2.equalizeHist(small)  # evens out the lighting - finds noticeably more faces
    min_side = max(24, int(min(small.shape) * 0.08))
    faces = _get_face_detector().detectMultiScale(small, scaleFactor=1.05, minNeighbors=3, minSize=(min_side, min_side))
    if len(faces) == 0:
        return img, False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # the largest face
    x, y, w, h = (v / scale for v in (x, y, w, h))
    pad_w, pad_h = w * margin, h * margin
    box = (
        max(0, int(x - pad_w)),
        max(0, int(y - pad_h)),
        min(img.width, int(x + w + pad_w)),
        min(img.height, int(y + h + pad_h)),
    )
    return img.crop(box), True


def center_square(img):
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


def prepare(lm, img, face_crop=None, square_crop=None):
    # Prepare an image exactly like the training images were prepared:
    #   1. resize it to the model's input size
    #   2. turn the pixels into numbers from 0 to 1
    #   3. make a batch of 1, the shape the model expects
    # Returns (batch, the resized image, face_found)
    face_crop = lm.spec.face_crop if face_crop is None else face_crop
    square_crop = lm.spec.square_crop if square_crop is None else square_crop

    face_found = None
    # Tiny images (like the 48x48 FER+ photos) are already face close-ups
    if face_crop and max(img.size) > 96:
        img, face_found = crop_face(img)
    if square_crop and not face_found:
        img = center_square(img)

    # Big photos are first shrunk smoothly to 128 px (the face mask training copies were made the same way),
    # so the final nearest-neighbour resize below doesn't turn them into noise
    if max(img.size) > 128:
        img = img.copy()
        img.thumbnail((128, 128), Image.LANCZOS)

    img = img.convert("L" if lm.color_mode == "grayscale" else "RGB")
    height, width = lm.image_size
    # Keras' load_img (used by the training data loader) resizes with nearest-neighbour by default
    resized = img.resize((width, height), Image.NEAREST)  # 1. resize

    pixels = np.asarray(resized, dtype="float32") / 255.0  # 2. normalize
    if lm.color_mode == "grayscale":
        pixels = pixels[..., np.newaxis]  # (48, 48) -> (48, 48, 1)
    batch = np.expand_dims(pixels, axis=0)  # 3. batch of 1
    return batch, resized, face_found


def predict_batch(lm, batch):
    with _predict_lock:
        return np.asarray(lm.model.predict_on_batch(batch))


def predict(lm, img, threshold=config.DEFAULT_THRESHOLD, face_crop=None, square_crop=None):
    batch, resized, face_found = prepare(lm, img, face_crop, square_crop)
    probabilities = predict_batch(lm, batch)[0]

    order = np.argsort(probabilities)[::-1]  # most likely first
    ranked = [(lm.classes[i], float(probabilities[i])) for i in order]
    class_name, confidence = ranked[0]
    is_unknown = confidence < threshold

    notes = []
    if face_found is False:
        notes.append("No face was found, so the whole image was used.")

    return Prediction(
        class_name=class_name,
        label="Unknown" if is_unknown else lm.spec.label(class_name),
        confidence=confidence,
        is_unknown=is_unknown,
        threshold=threshold,
        probabilities=ranked,
        model_input=resized,
        face_found=face_found,
        notes=notes,
    )


def list_images(folder, classes):
    # (path, class index) for every image in folder/<class>, like Keras' flow_from_directory
    items = []
    for index, class_name in enumerate(classes):
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _dirs, files in os.walk(class_dir):
            for file_name in sorted(files):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff")):
                    items.append((os.path.join(root, file_name), index))
    return items
