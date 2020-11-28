from Policy_Gradient.policy_gradient import Policy_gradient
from Policy_Gradient.model import model
from Games.Dots import *
import torch, os
import matplotlib.pyplot as plt

'''
    Vanila Policy Gradient method

    written by hanseul-jeong
    https://github.com/hanseul-jeong/RL
'''

inputSize = 5
hiddendim = 4
outputdim = 4
inputdim = 3
n_episode = 10000  # epoch
n_steps = 1024  # steps for 1 episode
lr = 1e-3
DISCOUNT_FACTOR = 0.99
gpu_number = 0  # in case of multi-gpu, choose gpu (e.g., 0, 1, 2)
SAVE = True
n_save = 100  # checkpoint episode

DEVICE = 'cuda:{0}'.format(gpu_number) if torch.cuda.is_available() else 'cpu'

# checkpoint
save_dir = "checkpoint"
if not os.path.isdir(save_dir) and SAVE:
    os.mkdir(save_dir)

# Target policy
policy = model(inputdim, hiddendim, outputdim, DEVICE).to(DEVICE)
optimizer = Policy_gradient(policy, inputSize, DISCOUNT_FACTOR, lr, device=DEVICE)

print("Let's get started !")
global_reward = []
for episode in range(1, n_episode + 1):
    # Game environment
    environment = gameEnv(inputSize)
    prev_state = environment.state

    for n_iter in range(1, n_steps + 1):
        action = optimizer.select_action(prev_state)
        next_state, imd_reward, done = environment.step(action)
        policy.rewards.append(imd_reward)
        prev_state = next_state
    global_reward.append(np.mean(policy.rewards))
    loss, reward = optimizer.update_policy()
    print(episode, ' is complete!')
    print('episode: {ep} reward: {reward}'.format(ep=episode, reward=global_reward[-1]))

    if SAVE and (episode % n_save == 0):
        torch.save(policy, os.path.join(save_dir, 'ck_{e}.pt'.format(e=episode // n_save)))

