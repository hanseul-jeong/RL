from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
n_RGB = 3

class Policy_gradient():
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

    def select_action(self, prev_state):
        X = torch.FloatTensor(prev_state).to(self.device).view(1, n_RGB, self.input_size + 2, self.input_size + 2)
        A = self.policy(X)
        action_dist = Categorical(A)
        action = action_dist.sample()
        if self.policy.policy_history.size(0) == 0:
            self.policy.policy_history = action_dist.log_prob(action)
        else:
            self.policy.policy_history = torch.cat([self.policy.policy_history, action_dist.log_prob(action)])
        return action

    # Calculate loss and backward
    def get_loss(self):
        reward_lists = []
        n_samples = len(self.policy.rewards)
        R = 0

        # Discounted reward
        for r in self.policy.rewards[::-1]:
            R = r + self.discount_factor * R
            reward_lists.insert(0, R)

        reward_list = torch.FloatTensor(reward_lists).to(self.device)

        # Normalization
        reward_list = (reward_list - reward_list.mean())/reward_list.std()

        loss = -torch.sum(self.policy.policy_history * reward_list) / n_samples
        return loss, reward_list

    # Backward loss and renewal previous policies and rewards
    def update_policy(self):
        loss, reward = self.get_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy.policy_history = Variable(torch.Tensor()).to(self.device)
        self.policy.rewards = []

        return loss, reward

