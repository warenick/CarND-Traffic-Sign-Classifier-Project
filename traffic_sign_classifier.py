import numpy as np
import matplotlib.pyplot as plt
import pickle

# load data

data_folder = "traffic-signs-data/"
training_file = data_folder+"train.p"
validation_file = data_folder+"valid.p"
testing_file = data_folder+"test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


import csv

csv_path = "signnames.csv"
reader = csv.reader(open(csv_path, "r"))
signnames = []
for row in reader:
    if "ClassId" in row:
        continue
    signnames.append(row)

n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(signnames)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validating examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

from pipeline_lib import *
imgs = []
titles = []
for i in range(n_classes):
    img_num = np.where(y_valid == i)  # return all labels of i class
    imgs.append(X_valid[img_num[0][0]])
    titles.append(signnames[i][0] + ' ' + signnames[i][1])
show_imgs_together(imgs,titles)

import cv2

for i in range(len(X_train)):
    cv2.normalize(X_train[i], X_train[i], 0, 255, norm_type=cv2.NORM_MINMAX)
for i in range(len(X_valid)):
    cv2.normalize(X_valid[i], X_valid[i], 0, 255, norm_type=cv2.NORM_MINMAX)
for i in range(len(X_test)):
    cv2.normalize(X_test[i], X_test[i], 0, 255, norm_type=cv2.NORM_MINMAX)

imgs = []
titles = []
for i in range(n_classes):
    img_num = np.where(y_valid == i)  # return all labels of i class
    imgs.append(X_valid[img_num[0][0]])
    titles.append(signnames[i][0] + ' ' + signnames[i][1])
# show_imgs_together(imgs,titles)

from Dataset import SignDataset
from torch.utils.data import DataLoader

train_dataset = SignDataset(X_train, y_train, n_classes)
test_dataset = SignDataset(X_test, y_test, n_classes)
valid_dataset = SignDataset(X_valid, y_valid, n_classes)
# print(test_dataset.__len__())
print(len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=10)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=10)
print(len(train_dataloader))

from LeNet import LeNet
net = LeNet()
print(net)
# some example https://neurohive.io/ru/tutorial/cnn-na-pytorch/

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
criterion = CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.005, momentum=0.9)
print(criterion)
print(optimizer)
# INIT_LR = 2*1e-3
# BS = 64
# Training
from torch import device
device = device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    net.train()
    train_loss = 0
    correct, total = 0, 0
    correct_test, total_test = 0, 0
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # if batch_idx % 100 == 99:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()
    print('Epoch: %d/%d' % (epoch+1,NUM_EPOCHS))
    print('Loss: %.3f Acc: %.3f %%' % ((train_loss),(100.*correct/total))) 
    correct, total = 0,0
    print('Accuracy test images: %d %%' % (
        100 * correct_test / total_test))
        # progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
        #              (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
print("finifh training")


# lets test
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)        
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
print('Accuracy of the network on train images: %d %%' % (
    100 * correct / total))
correct, total = 0,0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))
correct, total = 0,0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(valid_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
print('Accuracy of the network valid images: %d %%' % (
    100 * correct / total))
print(correct)
print(total)
print(net)