from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.autograd import Variable
import torch
import random
n_RGB = 3

class Policy_gradient():
    def __init__(self, policy, input_size, discount_factor, lr=1e-3, optimizer='Adam', device='cuda', policy2=None):
        self.policy = policy
        if policy2 is not None:
            self.policy_target = policy2
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.device = device
        self.n_acts = 4
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, prev_state, epsilon=None):
        X = torch.FloatTensor(prev_state).to(self.device).view(1, n_RGB, self.input_size + 2, self.input_size + 2)

        if epsilon is not None:
            p = random.random()
            if p <= epsilon:
                A = torch.FloatTensor([[1/self.n_acts for i in range(self.n_acts)]]).to(self.device)
            else:
                A = self.policy_target(X)
        else:
            A = self.policy_target(X)
        action_dist = Categorical(A)
        action = action_dist.sample()
        if self.policy.policy_history.dim() == 0:
            self.policy.policy_history = action_dist.log_prob(action)
        else:
            self.policy.policy_history = torch.cat([self.policy.policy_history, action_dist.log_prob(action)])
        return action

    def get_loss(self):
        reward_list = []
        R = 0
        for r in self.policy.rewards[::-1]:
            R = r + self.discount_factor * R
            reward_list.insert(0, R)
        reward_list = torch.FloatTensor(reward_list).to(self.device)
        n_samples = reward_list.size(0)
        loss = -torch.sum(self.policy.policy_history * reward_list)
        loss /= n_samples
        print(loss)
        return loss

    def update_policy(self):
        loss = self.get_loss()
        self.policy.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy.policy_history = Variable(torch.Tensor()).to(self.device)
        self.policy.rewards = []
        return loss