from Policy_Gradient.policy_gradient import Policy_gradient
import torch, os
import torch.optim as optim
'''
    Vanila Policy Gradient method

    written by hanseul-jeong
    https://github.com/hanseul-jeong/RL
'''

game = 'Dots'
if game in ['Dots','dots','dot']:
    from Policy_Gradient.model import Dots_model as model
elif game in ['CartPole', 'Cartpole', 'cartpole', 'cart', 'Cart', 'Pole', 'pole']:
    from Policy_Gradient.model import Cart_model as model

n_episode = 10000   # epoch
n_steps = 256       # steps for 1 episode
lr = 1e-4
DISCOUNT_FACTOR = 0.99
epsilon = 0.9
gpu_number = 1  # in case of multi-gpu, choose gpu (e.g., 0, 1, 2)
SAVE = True
n_save = 1000  # checkpoint episode
n_updates = 32

DEVICE = 'cuda:{0}'.format(gpu_number) if torch.cuda.is_available() else 'cpu'

# checkpoint
save_dir = "checkpoint"
if not os.path.isdir(save_dir) and SAVE:
    os.mkdir(save_dir)

# Target policy
policy = model(device=DEVICE).to(DEVICE)
PG = Policy_gradient(game, DISCOUNT_FACTOR, device=DEVICE)
optimizer = optim.Adam(policy.parameters(), lr=lr)

print("Let's get started !")
global_reward = []
for episode in range(1, n_episode + 1):
    # Game environment
    env, prev_state = PG.set_env()

    for n_iter in range(1, n_steps + 1):
        action = PG.select_action(policy, prev_state, epsilon)
        next_state, imd_reward, done, _ = env.step(action)
        policy.rewards.append(imd_reward)
        prev_state = next_state

        if done:
            break

    global_reward.append(sum(policy.rewards))
    loss = PG.get_loss(policy)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    PG.reset_policy(policy)

    print(episode, ' is complete!')
    print('episode: {ep} epsilon : {eps} reward: {reward}'.format(ep=episode, eps=epsilon, reward=global_reward[-1]))

    if episode % n_updates == 0:
        epsilon = 0.01 if episode < 0.01 else epsilon * 0.9

    if SAVE and (episode % n_save == 0):
        torch.save(policy, os.path.join(save_dir, 'ck_PG_dots_{e}.pt'.format(e=episode // n_save)))

