# 🌿 LeafRx — Plant Disease Detection Microservice

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-FF7C00)](https://www.gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

**LeafRx** is an end-to-end plant disease detection system that turns a fine-tuned deep-learning model into a usable application. Upload a leaf image through the web interface and the system sends it to a FastAPI backend, which preprocesses the image, runs inference with a ConvNeXt-Tiny model, and returns the predicted plant condition and confidence score.

> **The main goal of this repository is the application and its inference workflow.** The `train_model/` directory is included as supporting reference material for anyone who wants to understand how the model was trained, inspect the experiments, reproduce the training process, or try further improvements.

---

## ✨ What the application provides

- A Gradio frontend for uploading and analyzing leaf images.
- A FastAPI backend exposing a simple prediction endpoint.
- ConvNeXt-Tiny inference through PyTorch and Torchvision.
- Classification across the 38 PlantVillage classes supported by the trained model.
- Human-readable results that identify the plant, condition, healthy/diseased status, and prediction confidence.
- Dockerized frontend and backend services orchestrated with Docker Compose.
- CPU/GPU-aware inference: CUDA is used when available, otherwise inference falls back to CPU.

---

## 🧭 How it works

```text
┌────────────────────┐       image upload       ┌─────────────────────┐
│  Gradio Frontend   │ ───────────────────────▶ │  FastAPI Backend    │
│  Port 7860         │                          │  Port 8000          │
└────────────────────┘                          └──────────┬──────────┘
                                                           │
                                                           ▼
                                                ┌─────────────────────┐
                                                │ ConvNeXt-Tiny Model │
                                                │ Preprocessing +     │
                                                │ Classification      │
                                                └──────────┬──────────┘
                                                           │
                                                           ▼
                                                class + confidence
```

1. The user uploads a JPG or PNG leaf image in the Gradio interface.
2. The frontend sends the image to the backend's `/predict` endpoint.
3. The backend resizes and normalizes the image, then runs the trained model.
4. The API returns the predicted class and confidence.
5. The frontend presents the result as a plant condition with a confidence visualization.

---

## 🧱 Repository structure

```text
.
├── backend/
│   ├── app.py                  # FastAPI application and API endpoints
│   ├── model_service.py        # Model loading, preprocessing, and inference
│   ├── class_names.json        # Mapping of output indices to class names
│   ├── Dockerfile              # Backend container definition
│   └── requirements.txt         # Backend dependencies
├── frontend/
│   ├── frontend.py             # Gradio user interface
│   ├── Dockerfile              # Frontend container definition
│   └── requirements.txt         # Frontend dependencies
├── train_model/
│   ├── train.py                # Training entry point
│   ├── engine.py               # Training and validation loop
│   ├── prepare_dataset.py      # Dataset preparation and data loaders
│   ├── utils.py                # Training utilities
│   └── notebooks/
│       ├── Plant_Disease_Detection.ipynb
│       └── evaluate.ipynb      # Evaluation, plots, and example predictions
├── docker-compose.yml          # Runs the frontend and backend together
├── requirements.txt            # Root-level Python dependencies
└── README.md
```

---

## 🧪 Model and training reference

The deployed application uses a fine-tuned ConvNeXt-Tiny model with a 38-class output layer. The training code in `train_model/` is not required to run the application; it is kept for transparency, learning, and experimentation.

Use that folder if you want to:
- Review the dataset preparation pipeline.
- See how the classifier head was adapted for the PlantVillage classes.
- Inspect the training, validation, and checkpoint-saving logic.
- Re-run or modify the training workflow.
- Explore the notebooks and evaluate predictions on sample images.

The training notebook reports a PlantVillage dataset of 54,305 images across 38 classes. The model checkpoint used by the application is approximately 106 MB, according to the evaluation notebook.

---

## 📊 Evaluation results

The trained model achieved approximately 98% accuracy on the PlantVillage evaluation setup used for this project. The evaluation artifacts and visualizations are available in `train_model/notebooks/evaluate.ipynb`.

### Loss and accuracy curves

The evaluation notebook generates the training and validation curves for both loss and accuracy. Open the notebook to view the rendered plots and reproduce them from the recorded training results:
- View the loss and accuracy curves in `evaluate.ipynb`
- Open the complete training notebook

> The curves are included in the evaluation notebook because the experiment outputs are stored there. The application itself uses the resulting trained model for inference; it does not retrain the model at runtime.

---

## 🚀 Quick start with Docker Compose

Docker Compose is the recommended way to run the complete application because it starts the frontend and backend with the correct service-to-service configuration.

### 1. Clone the repository

```bash
git clone https://github.com/OwaisAbuSonaina/Plant-Disease-Detection-Microservice.git
cd Plant-Disease-Detection-Microservice
```

### 2. Start the services

```bash
docker compose up --build
```

Once the containers are running:
- **Frontend:** `http://localhost:7860`
- **Backend API:** `http://localhost:8000`
- **Interactive API documentation:** `http://localhost:8000/docs`

To stop the services:

```bash
docker compose down
```

---

## 🐍 Run the services without Docker

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend

In a second terminal:

```bash
cd frontend
pip install -r requirements.txt
python frontend.py
```

The frontend expects the backend at `http://0.0.0.0:8000/predict` by default. To use a different backend URL, set the `API_URL` environment variable before starting the frontend:

```bash
API_URL=http://localhost:8000/predict python frontend.py
```

---

## 📡 API reference

### `GET /`

Returns a simple health message confirming that the API is running.

### `POST /predict`

Accepts a leaf image as multipart form data.

**Accepted formats:** JPG and PNG

Example with cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/leaf-image.jpg"
```

Example response:

```json
{
  "class": "Tomato___Target_Spot",
  "confidence": 0.985
}
```

> The raw class name follows the PlantVillage naming convention. The frontend converts it into a more readable plant and condition display.

---

## ⚙️ Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `API_URL` | `http://0.0.0.0:8000/predict` | Backend prediction URL used by the frontend |

When running with Docker Compose, the frontend is configured to call the backend service using `http://backend:8000/predict`.

---

## ⚠️ Scope and limitations

- Predictions are intended for images similar to the training data and may not generalize to every camera, lighting condition, plant variety, or real-world disease presentation.
- A high confidence score does not guarantee a correct diagnosis.
- The system is a research and demonstration application, not a substitute for professional agricultural advice.
- For best results, provide a clear image containing a visible leaf with minimal obstruction.

---
