# ANN model

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device('cuda:0')
learning_rate = 0.01
train_ratio = 0.7
BATCH_SIZE = 10
epochs = 10


# data set
class CarDataset(Dataset):
    def __init__(self, csv_path, mode):
        self.data = pd.read_csv(csv_path) # ../是打开上级目录的文件，获取当前目录 path1 = os.path.abspath('.')
        self.mode = mode
        sep = int(train_ratio * len(self.data))
        if self.mode == 'train':
            self.inp = torch.tensor(self.data.iloc[:sep, :21].values.astype(np.float32))
            self.oup = torch.tensor(self.data.iloc[:sep, 21:].values.astype(np.float32))
        else:
            self.inp = torch.tensor(self.data.iloc[sep:, :21].values.astype(np.float32))
            self.oup = torch.tensor(self.data.iloc[sep:, 21:].values.astype(np.float32))

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.oup[idx]

        # inpt = torch.Tensor(self.inp[idx])
        # oupt = torch.Tensor(self.oup[idx])
        # return {'inp': inpt,
        #         'oup': oupt,
        #         }


dataset_train = CarDataset("car_onehot.csv", mode = 'train')
dataset_test = CarDataset("car_onehot.csv", mode = 'test')
data_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
print(data_train)
data_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

print(data_test)

# define net
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(21, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


net = Net()   #.to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
m = nn.Softmax(dim=1)
criterion = nn.MSELoss()


for epoch in range(epochs):
    for batch_idx,(data, target) in enumerate(data_train):
        output = net(data)
        output1 = m(output)
        loss = criterion(output1, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE, len(data_train),
                       100. * batch_idx / len(data_train), loss.item()))


    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(data_test):
        output = net(data)
        output1 = m(output)
        # print(output.size())
        test_loss += criterion(output1, target)

        # test_loss += criterion(logits, oup).item()
        # pred = output.data.max(1)[1]
        # print(target.data.size())
        # print(pred.size())
        # correct += pred.eq(target.data).sum()
        # only loss is calculated in this case

    test_loss /= len(data_test.dataset)    # calculate average loss
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

