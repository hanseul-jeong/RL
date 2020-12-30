from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import sys
import torch
import random, math

n_RGB = 3

class Actor_Critic():
    def __init__(self, game, adv, discount_factor=0.99, device='cuda',pixel=None):
        self.adv = adv
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

        elif game in ['Lunar', 'lunar']:
            self.game = 'Lunar'
            self.input_shape = (1, 8)
            self.n_acts = 4
        elif game in ['LunarCont', 'lunarcont']:
            self.game = 'LunarCont'
            self.input_shape = (1, 8)
            self.n_acts = 2
        else:
            print('Plz select game between dots, cartpole, lunar, and lunarcont ')
            sys.exit()
        print('{game} is selected !'.format(game=self.game))

        self.discount_factor = discount_factor
        self.device = device

    def set_env(self, seed=None):
        if self.game =='Dots':
            from Games.Dots import gameEnv
            env = gameEnv(self.pixel)
            init_state = env.state
        elif self.game == 'CartPole':
            import gym
            env = gym.make('CartPole-v1')
            init_state = env.reset()
        elif self.game == 'Lunar':
            import gym
            env = gym.make('LunarLander-v2')
            if not seed:
                env.seed(seed)
            init_state = env.reset()
        elif self.game == 'LunarCont':
            import gym
            env = gym.make('LunarLanderContinuous-v2')
            if not seed:
                env.seed(seed)
            init_state = env.reset()
        return env, init_state

    def select_action(self, actor, critic, prev_state, epsilon):
        X = torch.from_numpy(prev_state).float().to(self.device)
        X = X.view(self.input_shape)

        p = random.random()
        if p <= epsilon:
            if self.game != 'LumarCont':
                # uniform dist
                A = Variable(torch.FloatTensor([[1 / self.n_acts]*self.n_acts]), requires_grad=True).to(self.device)
            else:
                A = Variable(torch.FloatTensor([[0.0] * self.n_acts]), requires_grad=True).to(self.device)
        else:
            A = torch.softmax(actor(X), dim=-1)
        if self.game == 'LunarCont':
            var = torch.ones_like(A)
            action = torch.normal(A, var)
            logprob = self.get_logprob(action, A, var).unsqueeze(-1)
        else:
            action_dist = Categorical(A)
            action = action_dist.sample()
            logprob = action_dist.log_prob(action)

        if actor.policy_history.size(0) == 0:
            actor.policy_history = logprob
        else:
            actor.policy_history = torch.cat([actor.policy_history, logprob], dim=-1)

        if 'cuda' in self.device:
            action = action.detach().cpu()

        if self.game != 'Lunarcont':
            pred_q = critic(X)
            action = action.item()
        else:
            pred_q = critic(X)
            action = action.view(-1).numpy()
        if not self.adv:
            pred_q = pred_q[:, action]

        return action, pred_q.view(-1)  # remove batch dim

    # Calculate loss and backward
    def get_loss(self, actor, critic):
        reward_lists = []
        n_samples = len(actor.rewards)
        R = 0

        # Discounted reward
        for r in actor.rewards[::-1]:
            R = r + self.discount_factor * R
            reward_lists.insert(0, R)

        reward_list = torch.FloatTensor(reward_lists).to(self.device)

        # Normalization
        Q = torch.where(reward_list.std() !=0,
                                  (reward_list - reward_list.mean())/reward_list.std(),
                                  torch.zeros_like(reward_list))
        ######################################################### Check Lunar pred q ###########################################
        pred_q = torch.cat(critic.rewards, dim=-1)

        if self.adv:
            Q_ = (Q - pred_q).detach()
        else:
            Q_ = Q

        if self.game == 'LunarCont':
            loss_actor = -torch.sum(actor.policy_history * Q_)
            loss_critic = torch.sum((pred_q - Q)**2)
        else:
            loss_actor = -torch.sum(actor.policy_history * Q_) / n_samples
            loss_critic = torch.sum((pred_q - Q) ** 2)

        return loss_actor, loss_critic

    # Backward loss and renewal previous policies and rewards
    def reset_policy(self, policy):
        policy.policy_history = Variable(torch.Tensor()).to(self.device)
        policy.rewards = []

    def get_logprob(self, x, mu, sigma):
        '''
        return log-likelihood of x in case of N(mu, sigma^2)
        '''

        constant = 1 / (sigma * math.sqrt(2 * math.pi))
        z = -(x - mu) ** 2 / (2 * sigma.pow(2))
        prob = constant * torch.exp(z)

        return torch.log(prob)