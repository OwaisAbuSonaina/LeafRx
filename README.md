# 🌿 Plant Disease Detection Microservice

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container-2496ed?logo=docker)

## 📖 Project Overview

This project transitions a deep learning research model into a production-grade **Microservice**. It utilizes a fine-tuned **ConvNeXt-Tiny** model (98% Accuracy on PlantVillage) to classify plant diseases from leaf images.

The architecture is designed for **Scalability** and **Portability**:

1. **Inference Engine:** Decoupled prediction logic using a Singleton pattern.
2. **API Layer:** High-performance REST API built with **FastAPI**.
3. **Containerization:** Fully Dockerized environment for cloud-agnostic deployment.

---

## 🛠️ Tech Stack

- **Model:** ConvNeXt-Tiny (PyTorch)
- **Backend:** FastAPI (Python)
- **Containerization:** Docker
- **Deployment:** Hugging Face Spaces (Docker SDK)

---

## 📂 Repository Structure

```text
├── app/
│   ├── main.py              # API Gateway & Endpoints
│   ├── model_service.py     # Inference Logic & Preprocessing
│   └── class_names.json     # Class Label Mapping
├── models/
│   └── ConvNext-Tiny.pth    # Trained Model Artifact
├── Dockerfile               # Container Build Instructions
├── requirements.txt         # Dependencies
└── README.md                # Documentation

```

## 🚀 How to Run Locally

### Option 1: Using Docker (Recommended)

Build the image to ensure environment consistency:

```bash
docker build -t plant-disease-app .

```

Run the container:

```bash
docker run -p 8000:8000 plant-disease-app

```

Access the API at `http://localhost:8000/docs`

### Option 2: Manual Python Setup

Install dependencies:

```bash
pip install -r requirements.txt

```

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

```

---

## 📡 API Usage

**Endpoint:** `POST /predict`

**Request:** `multipart/form-data` (Image File)

**Response:**

```json
{
  "class": "Tomato___Target_Spot",
  "confidence": 0.985
}
```
