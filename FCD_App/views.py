# currency_detection_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .models import FusionModelImproved
from torchvision import transforms
from PIL import Image
import torch
import base64
import warnings

# Suppress the UserWarnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning)

# Load the model and move it to GPU
num_classes = 2
model_path = 'D:/Education/Final year Project/Fake Currency Detection/Model/fusion_model.pth'
fusion_model = FusionModelImproved(num_classes).cuda()
fusion_model.load_state_dict(torch.load(model_path))
fusion_model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image, model):
    # Ensure image has 3 channels
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def detect_currency(request):
    if request.method == 'POST' and request.FILES['imageInput']:
            image = request.FILES['imageInput']
            image_data = image.read()
            image.seek(0)
            img = Image.open(image)
            predicted_class = predict_image(img, fusion_model)
            result = "Fake Currency" if predicted_class == 0 else "Real Currency"
            # Convert image to Base64 for displaying in HTML
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return render(request, 'result.html', {'result': result, 'image_base64': image_base64})
    return render(request, 'index.html')
