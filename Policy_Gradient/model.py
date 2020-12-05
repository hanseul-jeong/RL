from torch.autograd import Variable
import torch.nn as nn
import torch

class Dots_model(nn.Module):
    def __init__(self, inputdim=3, n_actions=4, device='cuda'):
        super(Dots_model, self).__init__()
        self.inputdim = inputdim    # RGB
        self.outputdim = n_actions

        self.l1 = nn.Conv2d(self.inputdim, 32, 3, stride=1)
        self.b1 = nn.BatchNorm2d(32)
        self.l2 = nn.Conv2d(32, 64, 3, stride=1)
        self.b2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*3*3, 32)
        self.fc2 = nn.Linear(32, self.outputdim) # V(s), A(s,a)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.rewards = []

    def forward(self, x):
        z = torch.relu(self.b1(self.l1(x)))
        z = torch.relu(self.b2(self.l2(z)))
        z = z.view(-1, 64*3*3)
        z = torch.relu(self.fc1(z))
        z = torch.softmax(self.fc2(z), dim=-1)

        return z

class Cart_model(nn.Module):
    def __init__(self, inputdim=4, n_actions=2, device='cuda'):
        super(Cart_model, self).__init__()
        self.inputdim = inputdim
        self.outputdim = n_actions

        self.fc1 = nn.Linear(self.inputdim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, self.outputdim)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.rewards = []

    def forward(self, x):
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = torch.softmax(self.fc4(z), dim=-1)

        return z
