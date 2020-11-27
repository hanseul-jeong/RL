import torch.nn as nn
from torch.autograd import Variable
import torch

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
        self.fc1 = nn.Linear(16*3*3, self.outputdim)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.rewards = []

    def forward(self, x):
        z = torch.relu(self.l1(x))
        z = self.b1(z)
        z = torch.relu(self.l2(z))
        z = self.b2(z)
        z = z.view(-1, 16*3*3)
        z = torch.softmax(self.fc1(z), dim=-1)
        return z