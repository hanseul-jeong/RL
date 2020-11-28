from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
n_RGB = 3

class DQN():
    def __init__(self, policy, input_size, discount_factor, lr=1e-3, n_acts=4, optimizer='Adam', device='cuda'):
        self.policy = policy
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.device = device
        self.n_acts = n_acts    # # of actions

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
        else:
            print("Plz select optimizer in 'Adam' or 'SGD'")

    def select_action(self, prev_state, epsilon):
        X = torch.FloatTensor(prev_state).to(self.device).view(1, n_RGB, self.input_size + 2, self.input_size + 2)
        Qs = self.policy(X)
        p = random.random()
        if p < epsilon:
            action = random.randint(0,4)
        else:
            action = torch.argmax(Qs)
        if self.policy.policy_history.size(0) == 0:
            self.policy.policy_history = Qs[:,action]
        else:
            self.policy.policy_history = torch.cat([self.policy.policy_history, Qs[:,action]], dim=-1)
        return action

    # Calculate loss and backward
    def get_loss(self):
        Y = []
        n_samples = len(self.policy.rewards)
        R = 0

        # Discounted reward
        for r in self.policy.optimal_qs:
            R = r + self.discount_factor * R
            Y.insert(0, R)

        Y = torch.FloatTensor(Y).to(self.device)

        # Normalization
        Y = (Y - Y.mean())/Y.std()

        loss = torch.sum((self.policy.policy_history - Y)**2) / n_samples
        return loss, Y.mean()

    # Backward loss and renewal previous policies and rewards
    def update_policy(self):
        loss, reward = self.get_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy.policy_history = Variable(torch.Tensor()).to(self.device)
        self.policy.rewards = []
        self.policy.optimal_qs = []

        return loss, reward

