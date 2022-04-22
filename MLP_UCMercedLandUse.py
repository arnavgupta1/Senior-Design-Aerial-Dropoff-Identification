'''
DISCLAIMER: CURRENTLY FORMATTED TO WORK ON GOOGLE COLAB - WILL BE REFINED TO BE READABLE
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

from PIL import Image
import os
import cv2

#transform data (add to this later)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([256, 256]),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#import dataset
total_dataset = datasets.ImageFolder('drive/My Drive/UCMerced_LandUse/png_images', transform=transform) #change path to dataset
train_size = int(0.8 * len(total_dataset))
test_size = len(total_dataset) - train_size

#split dataset and load them into train and test
train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])
trainloader = DataLoader(dataset=train_dataset, batch_size=16)
testloader = DataLoader(dataset=test_dataset, batch_size=16)


# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21')
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 16 * 61 * 61, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 61 * 61)   # flatten features

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x[0]

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if we set the hardware to GPU in the Notebook settings, this should print a CUDA device:
print(device)

net.to(device)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        labels = labels.type(torch.FloatTensor)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#Testing the image sizes of the data
path = 'drive/My Drive/UCMerced_LandUse/png_images/'
for folder in os.listdir(path):
  folder_path = os.path.join(path, folder)
  for img in os.listdir(folder_path):
    im = cv2.imread(os.path.join(folder_path, img))
    if im.shape != (256, 256, 3):
      print(im)
      print(im.shape)

#More testing - seeing how transform affects the data
count = 0
for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    print(inputs.shape)
    print(labels.shape)
    count += 1
print(count)