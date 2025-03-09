import torch
from torchvision import transforms
from models.neural_network import DeepFakeDetector  
from PIL import Image
import torch.nn.functional as F
import os

MODEL_PATH = "backend/deepfake_model.pth"
model = DeepFakeDetector()

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),          
])

def predict_deepfake(image_path):
    if not os.path.exists(image_path):
        return None, "Error: Image file not found"

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = model(image)
        print("Raw model output:", output.item())

    deepfake_probability = torch.sigmoid(output).item()
    deepfake_detected = deepfake_probability > 0.5  

    return deepfake_detected, deepfake_probability
