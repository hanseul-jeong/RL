from Policy_Gradient.Policy_gradient import Policy_gradient
from Policy_Gradient.Model import model
import torch
import os
from Games.Dots import *
import matplotlib.pyplot as plt

inputSize = 14
hiddendim = 4
outputdim = 4
inputdim = 3
n_episode = 10000   # epoch
n_steps = 4096      # steps for 1 episode
n_batch = 256       # batch size
n_save = 10         # checkpoint episode
lr = 1e-3
UPDATE_STEP = 10    # update target policy
DISCOUNT_FACTOR = 0.99
epsilon = 0.1       # epsilon-greedy
n_selected = 5      # # of selective samples

window = plt.figure()
ax = window.add_subplot(1,1,1)
plt.xlabel('episode')
plt.ylabel('loss')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

save_dir = "checkpoint"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Target policy
policy = model(inputdim, hiddendim, outputdim, DEVICE).to(DEVICE)
optimizer = Policy_gradient(policy, inputSize, DISCOUNT_FACTOR, lr, device=DEVICE, n_selected=n_selected, epsilon=epsilon)

# Game environment
environment = gameEnv(inputSize)
prev_state = environment.state
print("Let's get started !")

for episode in range(1, n_episode+1):
    global_loss = []
    for n_iter in range(1, n_steps+1):
        action = optimizer.select_action(prev_state)
        next_state, imd_reward, done = environment.step(action)
        policy.rewards.append(imd_reward)

        # Backward loss
        if n_iter % n_batch == 0:
            loss = optimizer.update_policy()
            global_loss.append(loss)

        prev_state = next_state

    print(episode, ' is complete!')
    print('episode: {ep} loss: {loss}'.format(ep=episode, loss=np.mean(global_loss)))

    if episode % n_save == 0:
        torch.save(policy, os.path.join(save_dir, 'ck_{e}.pt'.format(e=episode//n_save)))