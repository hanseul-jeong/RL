import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
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
        self.l1 = nn.Conv2d(self.inputdim, 8, 3, stride=1)
        self.b1 = nn.BatchNorm2d(8)
        self.l2 = nn.Conv2d(8, 16, 3, stride=1)
        self.b2 = nn.BatchNorm2d(16)
        self.l3 = nn.Conv2d(16, 32, 3, stride=1)
        self.b3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*2*2, self.outputdim)

        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        # b_size = x.size(0)
        z = torch.relu(self.l1(x))
        z = self.b1(z)
        z = self.maxpool(z)
        z = torch.relu(self.l2(z))
        z = self.b2(z)
        z = self.maxpool(z)
        z = torch.relu(self.l3(z))
        z = self.b3(z)
        z = self.maxpool(z)
        z = z.view(-1, 32*2*2)
        z = torch.sigmoid(self.fc1(z))
        return z

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

inputSize = 30
hiddendim = 4
outputdim = 4
inputdim = 3
epoch = 50000
lr = 1e-4
UPDATE_STEP = 10
save_dir = "checkpoint"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

behavior_model = model(inputdim, hiddendim, outputdim)
target_model = model(inputdim, hiddendim, outputdim)
target_model.load_state_dict(behavior_model.state_dict())
target_model.eval()

optimizer = optim.Adam(behavior_model.parameters(), lr=lr)

def HuberLoss(delta):
    if delta < 1 and delta > -1:
        return (delta**2)/2
    return delta.abs() - 1/2

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

        if n_steps % UPDATE_STEP == 0:
            target_model.load_state_dict(behavior_model.state_dict())
            n_steps = 1
        else:
            n_steps += 1
    if e % 5 == 0:
        print('episode: {ep} loss: {loss}'.format(ep=e+1, loss=loss))
        torch.save(model, os.path.join(save_dir, 'ck_{e}.pt'.format(e=(e+1)//1000)))
