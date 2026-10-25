# 🔮 CNN Vision

Image classification with convolutional neural networks. Three models are trained in
`reference work.ipynb` and served by a Streamlit site and a FastAPI REST API, online through ngrok.

| Model | Classes | Test accuracy |
|---|---|---|
| 🍎 Fruit & Vegetable Recognition (Fruits 360) | Apple, Avocado, Banana, Eggplant, Tomato | 0.971 |
| 😷 Face Mask Detection | with mask, without mask | 0.946 |
| 😄 Facial Expression Recognition (FER+) | 8 expressions | 0.832 |

## Run it

```bash
pip install -r requirements.txt
python main.py
```

The first time, you are asked for your ngrok authtoken
(from https://dashboard.ngrok.com/get-started/your-authtoken). It is saved in ngrok's own config file, not in this project.
The terminal then shows the public address and a QR code for opening the site on a phone.

- `python main.py --no-ngrok`: run only on this computer (http://localhost:8765)
- `python main.py --reset-token`: enter a different ngrok authtoken

## What's inside

| File | What it does |
|---|---|
| `reference work.ipynb` | Stages 1-2: dataset, preprocessing, CNN, training, evaluation, saving the models |
| `main.py` | Starts everything: Streamlit site, REST API and the ngrok tunnel |
| `streamlit_app.py`, `app_pages/` | Stage 3: the site (classify, model statistics, API guide); desktop and phone layouts |
| `api.py` | FastAPI REST API under `/api` (docs at `/api/docs`); passes every other path on to the site |
| `vision/` | Shared code: loading the models, preparing images like in training, statistics, charts |
| `models/` | The trained models (`.pkl`), their class names and image settings (`.json`) and cached statistics |

## The site

1. **Choose a model**: fruits, face mask or facial expression.
2. **Add a photo** with the camera, from the phone's gallery, from a computer file, or pick a random test sample.
3. **💾 Save image** downloads the photo; **Classify** runs the CNN.

The photo is prepared exactly like the training images (resized to the model's input size, pixels scaled to 0-1,
shaped as a batch of one). The result shows the predicted class and the probability of every option.

**Bonuses**
- If the model is less than 60% sure, the answer is **Unknown** (the threshold can be changed in the settings).
- Online with **ngrok**, with a layout made for phones (picked automatically, or switch it in the settings).
- A **REST API** with model statistics and graphs.
- For facial expressions, the face is found and cropped first, so normal photos look like the training close-ups.

## REST API

| Method | Path | |
|---|---|---|
| GET | `/api/models` | The models, their classes and input size |
| POST | `/api/models/{model_id}/predict` | Classify an image (`file` form field; optional `threshold`, `face_crop`) |
| GET | `/api/stats` | Accuracy summary of all models |
| GET | `/api/models/{model_id}/stats` | Accuracy, precision / recall / F1 per class, confusion matrix, confidence, dataset, architecture |
| GET | `/api/models/{model_id}/stats/graphs/{graph}.png` | `per_class_metrics`, `confusion_matrix`, `class_distribution` |
| GET | `/api/models/{model_id}/stats/report` | The statistics as a web page with the graphs |

`model_id` is one of `fruits`, `face-mask` or `emotions`.

```bash
curl -F "file=@banana.jpg" http://localhost:8765/api/models/fruits/predict
```

Through a free ngrok address, add the header `ngrok-skip-browser-warning: 1` when calling the API from code.
