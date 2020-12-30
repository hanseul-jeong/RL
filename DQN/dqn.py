from torch.autograd import Variable
import torch.nn.functional as F
import torch
import random, sys
n_RGB = 3

class DQN():
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
            pass
        return env, init_state

    def select_action(self, policy_b, policy_t, prev_state, epsilon):
        X = torch.from_numpy(prev_state).float().to(self.device)
        X = X.view(self.input_shape)
        Qs = policy_b(X)
        opt_Qs = policy_t(X)

        p = random.random()
        if p < epsilon:
            action = random.randint(0,self.n_acts-1)
        else:
            action = torch.argmax(Qs).item()

        if policy_b.policy_history.size(0) == 0:
            policy_b.policy_history = Qs[:,action]
            policy_t.policy_history = torch.max(opt_Qs).unsqueeze(0)
        else:
            policy_b.policy_history = torch.cat([policy_b.policy_history, Qs[:,action]], dim=-1)
            policy_t.policy_history = torch.cat([policy_t.policy_history, torch.max(opt_Qs).unsqueeze(0)], dim=-1)

        return action

    # Calculate loss and backward
    def get_loss(self, policy_b, policy_t):
        # Return of next state
        optim_R = torch.cat([policy_t.policy_history[1:], torch.zeros(1).to(self.device)], dim=-1)

        # r + gamma * max_ R'
        Y = policy_b.rewards + (self.discount_factor * optim_R)

        # Normalization
        Y = (Y - Y.mean()) / (Y.std() + 1e-12)

        # loss = F.smooth_l1_loss(policy_b.policy_history, Y)
        loss = torch.sum((policy_b.policy_history - Y) ** 2)
        return loss, Y.sum()

    # Backward loss and renewal previous policies and rewards
    def reset_policy(self, policy_b, policy_t):
        policy_b.policy_history = Variable(torch.Tensor()).to(self.device)
        policy_b.rewards = Variable(torch.Tensor()).to(self.device)

        policy_t.policy_history = Variable(torch.Tensor()).to(self.device)


