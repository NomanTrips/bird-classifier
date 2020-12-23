import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
#from model import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app)

class_names = ['american-kestrel', 'bald-eagle', 'barred-owl', 'coopers-hawk', 'crow', 'great-horned-owl', 'non-hawk', 'northern-goshawk', 'osprey', 'peregrine-falcon', 'red-tailed-hawk', 'vulture']

#model = models.resnet18(pretrained=True)
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 10)
model = models.vgg16(pretrained=True)
model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names))
model.load_state_dict(torch.load('./vgg16_added_birds.pt', map_location=torch.device('cpu')))
model.eval()

#print(model)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, predicted = torch.max(outputs, 1)
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    output = m(outputs)
    predicted_class = class_names[predicted]
    return predicted_class


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        predicted_class = get_prediction(image_bytes=img_bytes)
        return jsonify({'class': predicted_class})


if __name__ == '__main__':
    app.run(host='0.0.0.0')