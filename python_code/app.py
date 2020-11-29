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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 24, 11)
        self.conv2 = nn.Conv2d(24, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(17424, 5808)  # 6*6 from image dimension
        self.fc2 = nn.Linear(5808, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


app = Flask(__name__)
CORS(app)
imagenet_class_index = json.load(open('./imagenet_class_index.json'))
#model = models.densenet121(pretrained=True)
model = Net()
#model = torch.load('./model.pt')
model.load_state_dict(torch.load('./model.pt'))
model.eval()



def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(150),
                                        transforms.CenterCrop(150),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    predicted = outputs.detach().cpu().numpy().squeeze(1)
    probability = 0
    for i, x in enumerate(predicted): # convert the sigmoid output to a 0 or 1 prediction
        if x > .5:
            probability = 1
        else:
            probability = 0
    print(probability)
    #_, y_hat = outputs.max(1)
    #predicted_idx = str(y_hat.item())
    #print(predicted_idx)
    if probability == 1:
            class_name = 'Hawk'
    else:
        class_name = 'Non-hawk'
    return class_name


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})


if __name__ == '__main__':
    app.run()