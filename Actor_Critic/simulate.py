from Policy_Gradient.policy_gradient import Policy_gradient
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import sys

'''
    Test policy gradient and save as .gif

    written by hanseul-jeong
    https://github.com/hanseul-jeong/RL
'''

class test_and_save():
    def __init__(self, game, file_path, seed, n_save=5):
        if game in ['Dots', 'dots', 'dot']:
            from Policy_Gradient.model import Dots_model as model
        elif game in ['CartPole', 'Cartpole', 'cartpole', 'cart', 'Cart', 'Pole', 'pole']:
            from Policy_Gradient.model import Cart_model as model
        elif game in ['Lunar', 'lunar']:
            from Policy_Gradient.model import Lunar_model as model
        else:
            sys.exit()

        max_steps = 1024  # steps for 1 episode
        DISCOUNT_FACTOR = 0.99
        epsilon = 0.0
        gpu_number = 1  # in case of multi-gpu, choose gpu (e.g., 0, 1, 2)
        DEVICE = 'cuda:{0}'.format(gpu_number) if torch.cuda.is_available() else 'cpu'

        # Target policy
        policy = model(device=DEVICE).to(DEVICE)
        file = torch.load(file_path)
        policy.load_state_dict(file.state_dict())
        PG = Policy_gradient(game, DISCOUNT_FACTOR, device=DEVICE)
        print("Let's get started !")
        # Game environment
        for idx in range(0, n_save):
            env, prev_state = PG.set_env(seed)
            figs = []
            window = plt.figure()

            for n_iter in range(1, max_steps + 1):
                action = PG.select_action(policy, prev_state, epsilon)
                next_state, imd_reward, done, _ = env.step(action)
                policy.rewards.append(imd_reward)
                prev_state = next_state
                figs.append([plt.imshow(env.render(mode='rgb_array'), interpolation=None)])
                if done:
                    break

            plt.title(sum(policy.rewards))
            anim = animation.ArtistAnimation(window, figs, interval=10)
            anim.save('{g}_seed_{s}_{i}.gif'.format(g=game, s=seed, i=idx), writer='imagemagick')
            plt.close()