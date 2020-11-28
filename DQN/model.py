from torch.autograd import Variable
import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self, inputdim, outputdim, device):
        super(model, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim

        self.l1 = nn.Conv2d(self.inputdim, 32, 3, stride=1)
        self.b1 = nn.BatchNorm2d(32)
        self.l2 = nn.Conv2d(32, 64, 3, stride=1)
        self.b2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*3*3, 32)
        self.fc2 = nn.Linear(32, self.outputdim) # V(s), A(s,a)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.optimal_qs = []
        self.rewards = []

    def forward(self, x):
        z = torch.relu(self.b1(self.l1(x)))
        z = torch.relu(self.b2(self.l2(z)))
        z = z.view(-1, 64*3*3)
        z = torch.relu(self.fc1(z))
        z = self.fc2(z)
        # V, A = torch.split(z, [1, self.outputdim], dim=-1)

        return z
