import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json


class PlantDiseaseInference:
    def __init__(self, model_path: str, class_names_path: str):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = models.convnext_tiny(weights=None)

        num_classes = 38
        self.model.classifier[2] = nn.Linear(
            self.model.classifier[2].in_features, num_classes
        )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = self.transform(image).unsqueeze(0)  
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        confidence, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_class_idx.item()]

        return {"class": predicted_class, "confidence": float(confidence.item())}
