from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import sys
import torch
import random
n_RGB = 3

class Policy_gradient():
    def __init__(self, game, discount_factor=0.99, device='cuda',pixel=None):
        if game in ['Dots','dots','dot']:
            self.game = 'Dots'
            if pixel is None:
                self.pixel = 5
            self.input_shape = (1, 3, self.pixel+2, self.pixel+2)     # padding
            # self.input_feature = 3          # RGB
            self.n_acts = 4    # # of actions

        elif game in ['CartPole', 'Cartpole', 'cartpole','cart','Cart','Pole','pole']:
            self.game = 'CartPole'
            self.input_shape = (1, 4)
            self.n_acts = 2
        else:
            print('Plz select game between dots and cartpole')
            sys.exit()
        print('{game} is selected !'.format(game=self.game))

        self.discount_factor = discount_factor
        self.device = device

    def set_env(self):
        if self.game =='Dots':
            from Games.Dots import gameEnv
            env = gameEnv(self.pixel)
            init_state = env.state
        elif self.game == 'CartPole':
            import gym
            env = gym.make('CartPole-v1')
            init_state = env.reset()
        return env, init_state

    def select_action(self, policy, prev_state, epsilon):
        X = torch.from_numpy(prev_state).float().to(self.device)
        X = X.view(self.input_shape)


        # p = random.random()
        # if p <= epsilon:
        #     A = torch.FloatTensor([[1 / self.n_acts for i in range(self.n_acts)]]).to(self.device)
        # else:
        A = policy(X)

        action_dist = Categorical(A)
        action = action_dist.sample()

        if policy.policy_history.size(0) == 0:
            policy.policy_history = action_dist.log_prob(action)
        else:
            policy.policy_history = torch.cat([policy.policy_history, action_dist.log_prob(action)], dim=-1)

        return action.item()

    # Calculate loss and backward
    def get_loss(self, policy):
        reward_lists = []
        n_samples = len(policy.rewards)
        R = 0

        # Discounted reward
        for r in policy.rewards[::-1]:
            R = r + self.discount_factor * R
            reward_lists.insert(0, R)

        reward_list = torch.FloatTensor(reward_lists).to(self.device)

        # Normalization
        reward_list = torch.where(reward_list.std() !=0,
                                  (reward_list - reward_list.mean())/reward_list.std(),
                                  torch.zeros_like(reward_list))

        loss = -torch.sum(policy.policy_history * reward_list) / n_samples

        return loss

    # Backward loss and renewal previous policies and rewards
    def reset_policy(self, policy):
        policy.policy_history = Variable(torch.Tensor()).to(self.device)
        policy.rewards = []