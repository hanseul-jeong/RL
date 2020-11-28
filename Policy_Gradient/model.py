from torch.autograd import Variable
import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self, inputdim, hiddendim, outputdim, device):
        super(model, self).__init__()
        self.inputdim = inputdim
        self.hiddendim = hiddendim
        self.outputdim = outputdim

        self.l1 = nn.Conv2d(self.inputdim, 32, 3, stride=1)
        self.b1 = nn.BatchNorm2d(32)
        self.l2 = nn.Conv2d(32, 64, 3, stride=1)
        self.b2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*3*3, 32)
        self.fc2 = nn.Linear(32, self.outputdim)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.rewards = []

    def forward(self, x):
        z = torch.relu(self.b1(self.l1(x)))
        z = torch.relu(self.b2(self.l2(z)))
        z = z.view(-1, 64*3*3)
        z = torch.relu(self.fc1(z))
        z = torch.softmax(self.fc2(z), dim=-1)

        return z
