# **Reinforcement Learning (RL)**

This projects apply basic RL concepts to various games with pytorch  
**More games (e.g., Baseball games) and RL methods (e.g., Double, Dueling DQN) will be updated soon.**
  
  
> ## Environment Settings

We need python 3.x, pytorch, numpy libraries.  
According to games types, gym and other libraries are needed.

The list of additional libraries  
- Dots      - (X)
- CartPole  - gym
- LunarLander - gym, box2d

the way of installation is below

    # 1) gym and Box2D
    pip install gym Box2D

> ## Games

1) Dots  
2) CartPole  
3) LunarLander
4) Baseball game (Coming Soon)

In case of OpenAI games, the goal of the each game is [here](https://github.com/openai/gym/wiki/Leaderboard, "learder board") and you can find more OpenAI games [here](https://gym.openai.com/envs/#classic_control, "OpenAI")

- ### **1) Dots**  

<p align='center'>
<img src="/img/GridWorld.png" width="400"/>
<p/>  

The agent (blue) moves to maximize score!    
There are 4 kinds of blocks which are blue, red, green, white.  
  
> **Blue** - Agent. we can only move this agent  
> **Red** - Obstacle. the score is decreased when the agent meet the obstacle  
> **Green** - Item. the score is increased when the agent meet the item  
> **White** - Edge. It represents the end of frame and the score is also decreased when the agent force to go to edge  

States are given with **colored map**, the actions are **4 (Up:0, Down:1, Left:2, Right:3)**  
You can control the size of map and the default value is 5x5 (total 7x7 with frame)
  
the original code of game is below
https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py

### Rewards Graph

<p align='center'>
<img src="/img/PG_reward_graph.png" width="400"    />
<p/>

### Agent Behaviors

<p align='center'>
<img src="/img/agent1.gif" width="250"    />
<img src="/img/agent2.gif" width="250"    />
<img src="/img/agent3.gif" width="250"   />
<p/>

- ### **2) CartPole**  

<p align='center'>
<img src="/img/CartPole.png" width="400"    />
<p/>  
We move cart to keep pole stand up within frame !  

States are given with **4 values**, the actions are **2 (left, right)** 


You can easily apply this game using gym library
    
    env = gym.make('cartpole-v1')

### Agent Behaviors

- #### Ver.1 1000 episode
<p align='center'>
<img src="/img/cartpole_1000.gif" width="350"    />
<img src="/img/cartpole_1000_v2.gif" width="350"    />
<p/>

- #### Ver.2 2000 episode
<p align='center'>
<img src="/img/cartpole_2000.gif" width="350"    />
<img src="/img/cartpole_2000_v2.gif" width="350"    />
<p/>

- #### Ver.1 4000 episode
<p align='center'>
<img src="/img/cartpole_4000.gif" width="350"    />
<img src="/img/cartpole_4000_v2.gif" width="350"    />
<p/>

- ### **3) LunarLander**

We aim to land the agent within two flags of moon surface.  
    
    env = gym.make('LunarLander-v2')    # discrete
    or
    env = gym.make('LunarLanderContinuous-v2')  # continuous

Discrete version : States and actions are given with **8 float values** and **4 integer values**, respectively.  
Continuous version : States and actions are given with **8 float values** and **2 float values (-1 ~ +1)**, respectively.    

### Agent Behaviors
    
## Ver.1 Discrete Actions  

<p align='center'>
<img src="/img/lunar_discrete.gif" width="350"    />
<p/>
    
## Ver.2 Continuous Actions  

<p align='center'>
<img src="/img/lunarcont.gif" width="350"    />
<p/>


> ## RL Methods

- ### **1) Policy gradient method**
Based on this game, the agent learned by vanila policy gradient and deep q learning.

- ### **2) Q learning**
Basic Q learning (without memory buffer)

- ### **3) A2C**
Actor-Critic with Advantage function

- ### **4) SAC**
Soft Actor-Critic


