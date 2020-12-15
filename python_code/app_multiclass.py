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

class_names = ['bald-eagle', 'barred-owl', 'coopers-hawk', 'crow', 'non-hawk', 'northern-goshawk', 'osprey', 'peregrine-falcon', 'red-tailed-hawk', 'vulture']

#model = models.resnet18(pretrained=True)
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 10)
model = models.vgg16(pretrained=True)
model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names))
model.load_state_dict(torch.load('./model_vgg16.pt', map_location=torch.device('cpu')))
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
    #probas = torch.exp(outputs).detach().numpy()
    #probas = np.squeeze(probas)
    #probas = torch.nn.softmax(outputs)
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    #print(outputs)
    output = m(outputs)
    #for x in outputs[0]:
     #   print(torch.nn.Softmax(x.item()))
    #print(torch.nn.Softmax(outputs))
    #outputs = torch.nn.functional.softmax(outputs, dim=0)
    #print(output)
    #print(probas)
    #for x in probas:
    #    print(x.shape)
    #    print('%f' % x)
    #print(torch.exp(outputs))
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
    app.run()