--- 
layout: default
title: Status
---

# Project Summary
<p>The goal of our project is to train our agent to play a modified Mincraft recreation of the game Pac-Man. The goal of our agent is to get as many points as possible before time runs out (500 steps). The agent will be placed in a 28 x 31 enclosed maze and will have to traverse the map in order to pick up diamonds located around the maze. The agent's score will be based on how many diamonds they collect in a given episode. There will be a total of 52 diamonds for the agent to collect. We will develop our AI using Malmo. </p>

<p>In this maze will also include a zombie, replacing the ghost in the original Pac-Man game. The agent will have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie and touches it, it will receive a penalty score.</p>

<p> </p>

<img src="https://user-images.githubusercontent.com/75513952/142336340-20a53401-44f9-48f4-a5fd-9d6d77205444.png" width="900" height="700">

### Environement Setup
- 28 x 31 Map
- 52 Diamonds

### Rewards
- Diamond +1
- Touching Zombie -1


# Approach
## Algorithm Used: PPO
<p>
One of the algorithms we are exploring is Proximal Policy Optimization or PPO for short. We used RLlib's implementation of a PPO trainer.
PPO is a on-policy algorithm, meaning that it explores by sampling actions according to the latest version of its stochastic policy. Essentially our agent learns from actions that it took using its current optimized policy and then updates its optimized policy. Initially the actions the agent will perform will be based on it's initial conditions and training procedure, but should get less random as more training goes on. Eventually this causes the agent to exploit already discovered rewards. 
</p>
  
We used discrete actions and defined the action space for PPO as follows:

### Action Space
- Move +1 -> Move 1 block forward
- Turn +1 -> Turn 90 degrees right
- Turn -1 -> Turn 90 degrees left

## Following diagram from RLlib's algorithm website 
<img src="https://user-images.githubusercontent.com/75513952/142348893-9389ccb9-e4f3-40da-83f1-b252248ae35c.png" width="800" height="300">





## Evaluation

## Remaining Goals and Challenges

<p> 
Since PPO is an on-policy algorithm, we would like to try an off-policy algorithm, such as Q-learning. Our remaining goal is to implement our Pac-man AI using tabular Q-Learning. 
</p>

### Resources Used 

- <https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html>
- <https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo>     List of RLlib algorithms
- <https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py>     RLlib's implementation of PPO
- <https://minecraft-archive.fandom.com/wiki/Blocks>     Minecraft wiki describing MineCraft block types
- <https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f#:~:text=On%2DPolicy%20learning%20algorithms%20are,already%20using%20for%20action%20selection.>


