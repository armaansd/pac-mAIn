--- 
layout: default
title: Status
---

## Project Summary
<p>The goal of our project is to train our agent to play a modified Mincraft recreation of the game Pac-Man. The goal of our agent is to get as many points as possible before time runs out (500 steps). The agent will be placed in a 28 x 31 enclosed maze and will have to traverse the map in order to pick up diamonds located around the maze. The agent's score will be based on how many diamonds they collect in a given episode. There will be a total of 52 diamonds for the agent to collect. We will develop our AI using Malmo. </p>

<p>In this maze will also include a zombie, replacing the ghost in the original Pac-Man game. The agent will have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie and touches it, it will receive a penalty score.</p>

<p> </p>

<img src="https://user-images.githubusercontent.com/75513952/142336340-20a53401-44f9-48f4-a5fd-9d6d77205444.png" width="800" height="500">

## Approach
Algorithm Used: PPO
- One of the algorithms we are exploring is Proximal Policy Optimization or PPO for short. We used RLlib's implementation of a PPO trainer. 
- We defined some parameters for PPO as follows:

### Environement Setup
- 28 x 31 Map
- 26 x 29 Walking space
- 52 Diamonds

### Rewards
- Diamond +1
- Touching Zombie -1

### Action Space
- Move +1 -> Move 1 block forward
- Turn +1 -> Turn 90 degrees right
- Turn -1 -> Turn 90 degrees left



## Evaluation

## Remaining Goals and Challenges

## Resources Used 

- <https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html>
- <https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo>     List of RLlib algorithms
- <https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py>     RLlib's implementation of PPO
- <https://minecraft-archive.fandom.com/wiki/Blocks>     Minecraft wiki describing MineCraft block types


