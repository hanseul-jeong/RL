import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Methods.utils import Policy_gradient
import matplotlib.pyplot as plt
import torch
import os
from Games.Dots import *
from torch.distributions.categorical import Categorical

class model(nn.Module):
    def __init__(self, inputdim, hiddendim, outputdim, device):
        super(model, self).__init__()
        self.inputdim = inputdim
        self.hiddendim = hiddendim
        self.outputdim = outputdim
        self.l1 = nn.Conv2d(self.inputdim, 8, 3, stride=2)
        self.b1 = nn.BatchNorm2d(8)
        self.l2 = nn.Conv2d(8, 16, 3, stride=2)
        self.b2 = nn.BatchNorm2d(16)
        # self.l3 = nn.Conv2d(16, 32, 3, stride=1)
        # self.b3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(16*3*3, self.outputdim)
        # self.maxpool = nn.MaxPool2d(2,2)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.rewards = []

    def forward(self, x):
        # b_size = x.size(0)
        z = torch.relu(self.l1(x))
        z = self.b1(z)
        # z = self.maxpool(z)
        z = torch.relu(self.l2(z))
        z = self.b2(z)
        # z = self.maxpool(z)
        # z = torch.relu(self.l3(z))
        # z = self.b3(z)
        # z = self.maxpool(z)
        # z = z.view(-1, 32*2*2)
        z = z.view(-1, 16*3*3)
        z = torch.softmax(self.fc1(z), dim=-1)
        return z

inputSize = 14
hiddendim = 4
outputdim = 4
inputdim = 3
n_episode = 50000
n_steps = 128
lr = 1e-3
UPDATE_STEP = 10
DISCOUNT_FACTOR = 0.99
epsilon = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = "checkpoint"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

policy = model(inputdim, hiddendim, outputdim, DEVICE).to(DEVICE)
policy_target = model(inputdim, hiddendim, outputdim, DEVICE).to(DEVICE)
optimizer = Policy_gradient(policy, inputSize, DISCOUNT_FACTOR, lr, device=DEVICE, policy2=policy_target)
policy_target.load_state_dict(policy.state_dict())
policy_target.eval()


environment = gameEnv(inputSize)
prev_state = environment.state
print("Let's get started !")
global_loss = 0

for episode in range(n_episode):
    # done = False
    state_history = []
    for n_iter in range(1, 10000):
        action = optimizer.select_action(prev_state, epsilon)
        next_state, imd_reward, done = environment.step(action)
        # print(imd_reward)
        policy.rewards.append(imd_reward)

        if n_iter % n_steps == 0 or done:
            loss = optimizer.update_policy()
            global_loss += loss
        if len(state_history) == 10:
            state_history.pop(-1)
        if n_iter % UPDATE_STEP == 0:
            policy_target.load_state_dict(policy.state_dict())
        state_history.insert(0, next_state)
        prev_state = next_state
    print(episode+1, ' is complete!')
    if episode % 10 == 0:
        print('episode: {ep} loss: {loss}'.format(ep=episode+1, loss=global_loss))
        torch.save(policy, os.path.join(save_dir, 'ck_{e}.pt'.format(e=(episode+1)//5)))
        global_loss = 0