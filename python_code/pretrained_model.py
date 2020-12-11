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
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load in each dataset and apply transformations using
# the torchvision datasets as datasets library

train_set = datasets.ImageFolder("hawk_data/train/", transform = transformations)
test_set = datasets.ImageFolder("hawk_data/test/", transform = transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
test_loader = torch.utils.data.DataLoader(test_set, batch_size =4, shuffle=True)

class_names = train_set.classes
print(class_names)

# functions to show an image
def imshow(img):
    img = img.cpu()
    #img = img / 2 + 0.5     # unnormalize ((image * std) + mean)
    img = ((img * .225) + .5) # unnormalize ((image * std) + mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
for x in range(1):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)

    # print labels
    print(labels)
    # show images
    imshow(torchvision.utils.make_grid(images))

criterion = nn.CrossEntropyLoss()
losses = []
minibatch_size = 4
m = len(train_set)

#model_ft = models.resnet18(pretrained=True)
model_ft = models.vgg16(pretrained=True)
#num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names))
model_ft = model_ft.to(device)


# Observe that all parameters are being optimized
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

def train_net():
    model_ft.train()
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
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
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
        labels_np = labels.detach().cpu().numpy()
        outputs = model_ft(images)
        _, preds = torch.max(outputs, 1)
        for i, x in enumerate(preds):
            total += 1
            correct += (class_names[preds[i]] == class_names[labels_np[i]])
    percent_correct = 100 * correct / total
    print('Accuracy of the network on the %d images: %d %%' %(total, percent_correct))

def show_predictions(data_loader):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)
    labels_np = labels.detach().cpu().numpy()
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %  class_names[labels_np[j]] for j in range(minibatch_size)))
    outputs = model_ft(images)
    _, preds = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % class_names[preds[j]] for j in range(minibatch_size)))

train_net()
predict(train_loader)
predict(test_loader)
show_predictions(test_loader)

torch.save(model_ft.state_dict(), 'model_vgg16.pt')
