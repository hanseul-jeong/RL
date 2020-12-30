from Actor_Critic.actor_critic import Actor_Critic
from Games.Dots import *
import torch, os
import torch.optim as optim
import matplotlib.pyplot as plt

'''
    Actor-Critic method

    written by hanseul-jeong
    https://github.com/hanseul-jeong/RL
'''

game = 'CartPole'

if game in ['Dots','dots','dot']:
    from Policy_Gradient.model import Dots_model as model
elif game in ['CartPole', 'Cartpole', 'cartpole', 'cart', 'Cart', 'Pole', 'pole']:
    from Policy_Gradient.model import Cart_model as model
elif game in ['Lunar', 'lunar']:
    from Policy_Gradient.model import Lunar_model as model

n_episode = 10000  # epoch
n_steps = 256  # steps for 1 episode
lr = 1e-4
DISCOUNT_FACTOR = 0.99
epsilon = 0.9
gpu_number = 1  # in case of multi-gpu, choose gpu (e.g., 0, 1, 2)
SAVE = True
n_save = 2000  # checkpoint episode
n_update = 128
DEVICE = 'cuda:{0}'.format(gpu_number) if torch.cuda.is_available() else 'cpu'
seed = 27
# checkpoint
save_dir = "checkpoint"
if not os.path.isdir(save_dir) and SAVE:
    os.mkdir(save_dir)

# Target policy
actor = model(device=DEVICE).to(DEVICE)
critic = model(device=DEVICE, n_actions=1).to(DEVICE)

AC = Actor_Critic(game, True, DISCOUNT_FACTOR, device=DEVICE)
optimizer_actor = optim.Adam(actor.parameters(), lr=lr)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr)

print("Let's get started !")
global_reward = []
for episode in range(1, n_episode + 1):

    env, prev_state = AC.set_env(seed)
    figs = []
    for n_iter in range(1, n_steps + 1):
        action, pred_q = AC.select_action(actor, critic, prev_state, epsilon)
        next_state, imd_reward, done, _ = env.step(action)
        actor.rewards.append(imd_reward)

        critic.rewards.append(pred_q)
        prev_state = next_state
        if done:
            break

    global_reward.append(sum(actor.rewards))
    loss_actor, loss_critic = AC.get_loss(actor, critic)

    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()
    AC.reset_policy(actor)

    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()
    AC.reset_policy(critic)

    print(episode, ' is complete!')
    print('episode: {ep} epsilon {eps:.4f} reward: {reward:.4f}'.format(ep=episode, eps=epsilon, reward=global_reward[-1]))

    if (episode % n_update) == 0:
        epsilon = 0.01 if episode < 0.01 else epsilon * 0.9


plt.close()