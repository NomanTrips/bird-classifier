import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Specify transforms using torchvision.transforms as transforms
transformations = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.225, .225, .225])
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load in each dataset and apply transformations using
# the torchvision datasets as datasets library

train_set = datasets.ImageFolder("data/train/", transform = transformations)
test_set = datasets.ImageFolder("data/test/", transform = transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
test_loader = torch.utils.data.DataLoader(test_set, batch_size =4, shuffle=True)

classes = [0,1]

# functions to show an image
def imshow(img):
    img = img.cpu()
    #img = img / 2 + 0.5     # unnormalize ((image * std) + mean)
    img = ((img * .225) + .5) # unnormalize ((image * std) + mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)

# print labels
print(labels)
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# show images
imshow(torchvision.utils.make_grid(images))

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

net = Net()
net.to(device)
#print(net)

criterion = nn.BCELoss()#nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
losses = []
minibatch_size = 4
m = len(train_set)

# Print model's state_dict
#print("Model's state_dict:")
#for param_tensor in net.state_dict():
#    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# Print optimizer's state_dict
#print("Optimizer's state_dict:")
#for var_name in optimizer.state_dict():
#    print(var_name, "\t", optimizer.state_dict()[var_name])

def train_net():
    for epoch in range(30):  # loop over the dataset multiple times
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).squeeze(1)
            #print(outputs)
            #print(labels.float())
            loss = criterion(outputs, labels.float()) # labels
            loss.backward()
            minibatch_cost += loss.item() / num_minibatches
            optimizer.step()

        if epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if epoch % 1 == 0:
            losses.append(minibatch_cost)
    print('Finished Training')
    # plot the cost
    plt.plot(np.squeeze(losses))
    plt.ylabel('loss')
    plt.xlabel('iterations (per tens)')
    plt.title("placeholder")
    #plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def predict(data_loader):
    correct = 0
    total = 0
    for i, data in enumerate(data_loader, 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        predicted = outputs.detach().cpu().numpy().squeeze(1) # put the output predictions in a 1D numpy array
        labels_np = labels.detach().cpu().numpy()
        probability = 0
        for i, x in enumerate(predicted): # convert the sigmoid output to a 0 or 1 prediction
            if x > .5:
                probability = 1
            else:
                probability = 0
            total += 1
            correct += (probability == labels_np[i])
    percent_correct = 100 * correct / total
    print('Accuracy of the network on the %d images: %d %%' %(total, percent_correct))

def show_predictions(data_loader):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j].detach().cpu().numpy() for j in range(minibatch_size)))
    outputs = net(images)
    predicted = outputs.detach().cpu().numpy().squeeze(1) # put the output predictions in a 1D numpy array
    probas = np.array([])
    for i, x in enumerate(predicted): # convert the sigmoid output to a 0 or 1 prediction
        if x > .5:
            probas = np.append(probas, 1)
        else:
            probas = np.append(probas, 0)
    print('Predicted: ', ' '.join('%5s' % int(probas[j]) for j in range(minibatch_size)))

train_net()
predict(train_loader)
predict(test_loader)
show_predictions(test_loader)

torch.save(net.state_dict(), 'model.pt')
