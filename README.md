# **Reinforcement Learning (RL)**

This projects apply basic RL concepts to various games.  

**1) GridWorld 2) CartPole 3) Baseball game (Coming soon)**


**More games (e.g., Baseball games) and RL methods (e.g., Double, Dueling DQN) will be updated soon.**
  

- ### **GridWorld**  

<p align='center'>
<img src="/img/GridWorld.png" width="400"    />
<p/>  
The agent (blue) moves to maximize score !
There are 4 kinds of blocks which are blue, red, green, white.
  
**Blue** - Agent. we can only move this agent  
**Red** - Obstacle. the score is decreased when the agent meet the obstacle  
**Green** - Item. the score is increased when the agent meet the item  
**White** - Edge. It represents the end of frame and the score is also decreased when the agent force to go to edge  

States are given with colored map, the actions are 4 (Up:0, Down:1, Left:2, Right:3)
  
the original code of game is below
https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py

Based on this game, the agent learned by vanila policy gradient and deep q learning.
> ## **policy gradient method**

### Rewards graph

<p align='center'>
<img src="/img/PG_reward_graph.png" width="400"    />
<p/>

### Agent Behaviors

<p align='center'>
<img src="/img/agent1.gif" width="250"    />
<img src="/img/agent2.gif" width="250"    />
<img src="/img/agent3.gif" width="250"   />
<p/>

- ### **CartPole**  

<p align='center'>
<img src="/img/CartPole.png" width="400"    />
<p/>  
We move cart to keep pole stand up within frame !
States are given with 4 values, the actions are 2 (left, right) 


You can easily apply this game using gym library

    pip install gym

> ## **policy gradient method**
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