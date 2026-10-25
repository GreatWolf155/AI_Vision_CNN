"""The three trained models, where their files live, and the site's colors."""

import os
from dataclasses import dataclass, field

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
# A few test images per class, kept in the repo so the "Sample" button works without the datasets
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")

# Bonus: predictions below this confidence are shown as "Unknown"
DEFAULT_THRESHOLD = 0.60


@dataclass(frozen=True)
class ModelSpec:
    id: str                 # used in URLs, e.g. /api/models/fruits
    name: str               # the notebook's name for it - also the .pkl / .json file name
    title: str              # page heading
    icon: str
    tagline: str
    test_dir: str           # test images, one folder per class (same folders the notebook evaluated on)
    train_dir: str
    labels: dict = field(default_factory=dict)  # class folder name -> friendly label
    face_crop: bool = False      # find the face and crop to it before predicting
    square_crop: bool = False    # center-crop to a square before predicting
    hint: str = ""

    @property
    def model_path(self):
        return os.path.join(MODELS_DIR, f"{self.name}.pkl")

    @property
    def info_path(self):
        return os.path.join(MODELS_DIR, f"{self.name}.json")

    @property
    def samples_dir(self):
        return os.path.join(SAMPLES_DIR, self.id)

    @property
    def stats_path(self):
        return os.path.join(MODELS_DIR, f"{self.name}.stats.json")

    def label(self, class_name):
        return self.labels.get(class_name, class_name)


MODELS = {
    "fruits": ModelSpec(
        id="fruits",
        name="Fruit and Vegetable Recognition",
        title="Fruit & Vegetable Recognition",
        icon="🍎",
        tagline="Recognizes apples, avocados, bananas, eggplants and tomatoes",
        test_dir=os.path.join(DATA_DIR, "fruits", "fruits-subset", "Test"),
        train_dir=os.path.join(DATA_DIR, "fruits", "fruits-subset", "Training"),
        labels={
            "Apple": "🍎 Apple",
            "Avocado": "🥑 Avocado",
            "Banana": "🍌 Banana",
            "Eggplant": "🍆 Eggplant",
            "Tomato": "🍅 Tomato",
        },
        # Fruits 360 photos are square, with the fruit in the middle
        square_crop=True,
        hint="Works best with one fruit in the middle of the photo on a plain, light background.",
    ),
    "face-mask": ModelSpec(
        id="face-mask",
        name="Face Mask Detection",
        title="Face Mask Detection",
        icon="😷",
        tagline="Tells whether the person in the photo is wearing a mask",
        test_dir=os.path.join(DATA_DIR, "face-mask-detection", "FMD_SPLIT", "test"),
        train_dir=os.path.join(DATA_DIR, "face-mask-detection", "FMD_SPLIT", "train"),
        labels={
            "with_mask": "😷 With mask",
            "without_mask": "🙂 Without mask",
        },
        hint="Works best when the face fills most of the photo.",
    ),
    "emotions": ModelSpec(
        id="emotions",
        name="Facial Expression Recognition",
        title="Facial Expression Recognition",
        icon="😄",
        tagline="Recognizes 8 facial expressions (FER+)",
        test_dir=os.path.join(DATA_DIR, "fer2013plus", "fer2013plus", "fer2013", "test"),
        train_dir=os.path.join(DATA_DIR, "fer2013plus", "fer2013plus", "fer2013", "train"),
        labels={
            "anger": "😠 Anger",
            "contempt": "😒 Contempt",
            "disgust": "🤢 Disgust",
            "fear": "😨 Fear",
            "happiness": "😄 Happiness",
            "neutral": "😐 Neutral",
            "sadness": "😢 Sadness",
            "surprise": "😲 Surprise",
        },
        # FER+ images are tightly cropped faces, so a full photo needs the face cut out first
        face_crop=True,
        hint="Look straight at the camera - the face is found and cropped automatically.",
    ),
}

DEFAULT_MODEL = "fruits"

# Site colors (blue -> purple)
BLUE = "#3b5bdb"
INDIGO = "#4c3fd1"
PURPLE = "#7c3aed"
INK = "#1e1b4b"
INK_MUTED = "#5b5880"
SURFACE = "#fcfcff"
GRID = "#e6e4f5"

# Chart colors (validated with the dataviz palette checker: violet / orange / aqua pass CVD checks in this order)
SERIES = ["#6250d6", "#eb6834", "#1baf7a"]
MUTED_BAR = "#c9c6e3"
# Sequential blue ramp (light -> dark) for the confusion matrix
BLUE_RAMP = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
