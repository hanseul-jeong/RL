from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch
import random
n_RGB = 3

class Policy_gradient():
    def __init__(self, policy, input_size, discount_factor, lr=1e-3, n_acts=4, optimizer='Adam', device='cuda', n_selected=-1, epsilon=-1):
        self.policy = policy
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.device = device
        self.n_acts = n_acts    # # of actions
        self.n_selected = n_selected
        self.epsilon = epsilon
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, prev_state):
        X = torch.FloatTensor(prev_state).to(self.device).view(1, n_RGB, self.input_size + 2, self.input_size + 2)

        if self.epsilon != -1:
            p = random.random()
            if p <= self.epsilon:
                A = torch.FloatTensor([[1/self.n_acts for i in range(self.n_acts)]]).to(self.device)
            else:
                A = self.policy_target(X)
        else:
            A = self.policy_target(X)
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
        # backward with selective samples
        if self.n_selected != -1:
            # Select best and worst score
            indices_ = np.argsort(reward_lists)
            indices = np.concatenate([indices_[:self.n_selected], indices_[-self.n_selected:]], axis=0)
            mask = torch.zeros(n_samples).to(self.device)
            for idx in indices:
                mask[idx] = 1
            reward_list = reward_list * mask
            n_samples = self.n_selected * 2
        # Normalization
        reward_list = (reward_list - reward_list.mean())/reward_list.std()

        loss = -torch.sum(self.policy.policy_history * reward_list) / n_samples
        return loss

    # Backward loss and renewal previous policies and rewards
    def update_policy(self):
        loss = self.get_loss()
        self.policy.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy.policy_history = Variable(torch.Tensor()).to(self.device)
        self.policy.rewards = []
        return loss