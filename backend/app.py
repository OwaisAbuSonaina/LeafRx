# app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from model_service import PlantDiseaseInference
import uvicorn
from huggingface_hub import hf_hub_download
import os
from pathlib import Path

app = FastAPI(title="Plant Disease Detection Microservice")

CLASSES_PATH = "class_names.json"
inference_engine = None

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_FILENAME = "ConvNeXt-tiny.pth"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

if not MODEL_PATH.is_file():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="OwaisAbuSonaina/LeafRx-ConvNeXt-tiny",
        filename=MODEL_FILENAME,
        local_dir=str(MODEL_DIR)
    )


@app.on_event("startup")
def load_model():
    global inference_engine
    try:
        inference_engine = PlantDiseaseInference(MODEL_PATH, CLASSES_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
def home():
    return {"message": "Plant Disease API is running. Use /predict to classify images."}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File must be JPG or PNG")

    try:
        image_bytes = await file.read()
        prediction = inference_engine.predict(image_bytes)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
