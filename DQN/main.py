from DQN.dqn import DQN
import torch.optim as optim
import torch, os

'''
    Vanila Deep Q network method (w/o replay buffer and epsilon-greedy)

    written by hanseul-jeong
    https://github.com/hanseul-jeong/RL
'''

game = 'Dots'
if game in ['Dots','dots','dot']:
    from DQN.model import Dots_model as model
elif game in ['CartPole', 'Cartpole', 'cartpole', 'cart', 'Cart', 'Pole', 'pole']:
    from DQN.model import Cart_model as model

n_episode = 10000  # epoch
n_steps = 256  # steps for 1 episode
n_updates = 64

lr = 1e-4
DISCOUNT_FACTOR = 0.99
epsilon = 0.9   # epsilon-greedy (-1 : Only greedy)
gpu_number = 0  # in case of multi-gpu, choose gpu (e.g., 0, 1, 2)
SAVE = True
n_save = 1000  # checkpoint episode

DEVICE = 'cuda:{0}'.format(gpu_number) if torch.cuda.is_available() else 'cpu'

# checkpoint
save_dir = "checkpoint"
if not os.path.isdir(save_dir) and SAVE:
    os.mkdir(save_dir)

# Target policy
policy_b = model(device=DEVICE).to(DEVICE)
policy_t = model(device=DEVICE).to(DEVICE)
policy_t.load_state_dict(policy_b.state_dict())
policy_t.eval()

dqn = DQN(game, DISCOUNT_FACTOR, device=DEVICE)
optimizer = optim.Adam(policy_b.parameters(), lr=lr)

print("Let's get started !")
global_reward = []
for episode in range(1, n_episode + 1):
    # Game environment
    env, prev_state = dqn.set_env()
    for n_iter in range(1, n_steps + 1):
        # while not done:

        action = dqn.select_action(policy_b, policy_t, prev_state, epsilon)
        # next_state, imd_reward, done = environment.step(action)
        next_state, imd_reward, done, _ = env.step(action)

        if policy_b.rewards.size(0) == 0:
            policy_b.rewards = torch.FloatTensor([imd_reward]).to(DEVICE)
        else:
            policy_b.rewards = torch.cat([policy_b.rewards, torch.FloatTensor([imd_reward]).to(DEVICE)], dim=-1)
        prev_state = next_state
        if done:
            break

    global_reward.append(policy_b.rewards.sum().item())
    loss, optimal_Y = dqn.get_loss(policy_b, policy_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dqn.reset_policy(policy_b, policy_t)

    if episode % n_updates == 0:
        policy_t.load_state_dict(policy_b.state_dict())
        epsilon = 0.01 if episode < 0.01 else epsilon * 0.9

    print(episode, ' is complete!')
    print('episode: {ep} epsilon {eps:.4f} reward: {reward}'.format(ep=episode, eps=epsilon, reward=global_reward[-1]))

    if SAVE and (episode % n_save == 0):
        torch.save(policy_t, os.path.join(save_dir, 'ck_dqn_dots_{e}.pt'.format(e=episode // n_save)))

