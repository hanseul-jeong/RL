from dqn.DQN import DQN
from dqn.model import model
from Games.Dots import *
import torch, os
import matplotlib.pyplot as plt

'''
    Vanila Deep Q network method (w/o replay buffer and epsilon-greedy)

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
epsilon = 0.1   # epsilon-greedy (-1 : Only greedy)
gpu_number = 0  # in case of multi-gpu, choose gpu (e.g., 0, 1, 2)
SAVE = False
n_save = 100  # checkpoint episode
n_acts = 4

DEVICE = 'cuda:{0}'.format(gpu_number) if torch.cuda.is_available() else 'cpu'

# checkpoint
save_dir = "checkpoint"
if not os.path.isdir(save_dir) and SAVE:
    os.mkdir(save_dir)

# Target policy
policy = model(inputdim, outputdim, DEVICE).to(DEVICE)
optimizer = DQN(policy, inputSize, DISCOUNT_FACTOR, lr, device=DEVICE)

window = plt.figure()
ax = window.add_subplot(1,1,1)
plt.xlabel('episode', fontsize=17)
plt.ylabel('Average rewards', fontsize=17)

print("Let's get started !")
global_reward = []
for episode in range(1, n_episode + 1):
    # Game environment
    environment = gameEnv(inputSize)
    prev_state = environment.state

    for n_iter in range(1, n_steps + 1):
        action = optimizer.select_action(prev_state, epsilon)
        next_state, imd_reward, done = environment.step(action)
        policy.rewards.append(imd_reward)
        max_r = -100
        for a in range(n_acts):
            environment.state = prev_state
            _, r, _ = environment.step(a)
            if max_r < r:
                max_r = r
        policy.optimal_qs.append(max_r)
        prev_state = next_state

    global_reward.append(np.mean(policy.rewards))
    loss, optimal_Y = optimizer.update_policy()
    print(episode, ' is complete!')
    print('episode: {ep} loss: {loss}'.format(ep=episode, loss=loss.item()))

    ax.plot(global_reward, color='gray', linewidth=0.5)
    plt.pause(0.0000001)
    ax.lines.pop()


    if SAVE and (episode % n_save == 0):
        torch.save(policy, os.path.join(save_dir, 'ck_{e}.pt'.format(e=episode // n_save)))

plt.close()

