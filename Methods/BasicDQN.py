import torch.nn as nn
import torch.optim as optim
import torch
import os
from Games.Dots import *
from torch.distributions.categorical import Categorical

class model(nn.Module):
    def __init__(self, inputdim, hiddendim, outputdim):
        super(model, self).__init__()
        self.inputdim = inputdim
        self.hiddendim = hiddendim
        self.outputdim = outputdim
        self.l1 = nn.Conv2d(self.inputdim, 64, 3, stride=1)
        self.l2 = nn.Conv2d(64, 128, 3, stride=1)
        self.l3 = nn.Conv2d(128, 256, 3, stride=1)
        self.fc1 = nn.Linear(256*2*2, self.outputdim)

        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        # b_size = x.size(0)
        z = torch.tanh(self.l1(x))
        z = self.maxpool(z)
        z = torch.tanh(self.l2(z))
        z = self.maxpool(z)
        z = torch.tanh(self.l3(z))
        z = self.maxpool(z)
        z = z.view(-1, 256*2*2)
        z = torch.sigmoid(self.fc1(z))
        return z
inputSize = 30
hiddendim = 4
outputdim = 4
inputdim = 3
epoch = 50000
lr = 1e-4
save_dir = "checkpoint"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

model = model(inputdim, hiddendim, outputdim)
optimizer = optim.Adam(model.parameters(), lr=lr)

environment = gameEnv(inputSize)
action = 1
print("Let's get started !")
for e in range(epoch):
    done = False
    n_steps = 1
    while not done:
        state, score, done = environment.step(action)
        X = torch.Tensor(state).view(1, 3, inputSize+2, inputSize+2)
        A = model(X)
        action_dist = Categorical(A)
        action = action_dist.sample()
        loss = -action_dist.log_prob(action)*score

        model.zero_grad()
        loss.backward()
        optimizer.step()
        n_steps += 1
    if e % 5 == 0:
        print('episode: {ep} loss: {loss}'.format(ep=e+1, loss=loss))
        torch.save(model, os.path.join(save_dir, 'ck_{e}.pt'.format(e=(e+1)//1000)))
