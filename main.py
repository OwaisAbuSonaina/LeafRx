from fastapi import FastAPI, UploadFile, File, HTTPException
from backendapp.model_service import PlantDiseaseInference
import uvicorn

app = FastAPI(title="Plant Disease Detection Microservice")

MODEL_PATH = "models/ConvNext-Tiny.pth"
CLASSES_PATH = "app/class_names.json"
inference_engine = None


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
