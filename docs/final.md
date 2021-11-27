--- 
layout: default
title: final
---
This will be exchanged for final vid later
<iframe width="560" height="315" src="https://www.youtube.com/embed/GXTNISa5NKk" frameborder="0" allowfullscreen>
</iframe>

      
# Project Summary
<p>The goal of our project is to train our agent to play a modified recreation of Pac-Man on Minecraft. The goal of our agent is to obtain the highest score for the given map. The pellets from the original game will be substituted with diamonds. The agent's score will be based on how many diamonds they collect in a given episode. There will be a total of 35 diamonds for the agent to collect. We will develop our AI using Malmo. </p>

<p>In this maze will also include a zombie, replacing the ghost in the original Pac-Man game. The agent will have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie it will receive a penalty score. If it gets attacked by the zombie, agent will "die" and received a penalty score.</p>

<p>The goal of this project is to implement the environment ourselves and implement and compare more than one algorithm. </p>


### Environement Setup
- 21 x 21 Map
- 35 Diamonds
<img src="https://user-images.githubusercontent.com/75513952/143675924-3145d205-3c8a-4f85-818c-735aa89c3387.png" width="300" height="300">


### Rewards
- Diamond +1
- Near Zombie -1
- Damage by Zombie -5
- Touching wall -10

# Approach
## Algorithm Used: PPO
<p>One of the algorithms we are exploring is Proximal Policy Optimization or PPO for short. We used RLlib's implementation of a PPO trainer.
PPO is a on-policy algorithm, meaning that it explores by sampling actions based on its latest version of its stochastic policy. Essentially our agent learns from actions that it took using its current optimized policy and then updates its optimized policy. Initially the actions the agent will perform will be based on it's initial conditions and training procedure, but should get less random as more training goes on. Eventually this causes the agent to exploit already discovered rewards. </p>

<p>In our scenario, an 2 x 25 x 25 observation grid for entities near the agent will be generated. Diamonds will be enumerated with the value of 1. Zombies will be enumerated with a value of -1. Since diamonds will be removed from the map whenever the agent picks it up, an observation grid for entities is needed instead of for blocks around the agent. Adjusting the index of the item according to the agent will help update the agent's obervation, reward, and states to update its next policy.</p>
  
We used discrete actions and defined the action space for PPO as follows:

### Action Space
- Move +1 -> Move 1 block forward
- Turn +1 -> Turn 90 degrees right
- Turn -1 -> Turn 90 degrees left

## Diagram of PPO architecture from RLlib's algorithm website 
<img src="https://user-images.githubusercontent.com/75513952/142348893-9389ccb9-e4f3-40da-83f1-b252248ae35c.png" width="800" height="300">

## PPO defines a probability ratio between its new policy and old policy
- r(θ) = π<sub>θ</sub>(a given s) / π<sub>θold</sub>(a given s)

## Objective function of PPO
<img src="https://user-images.githubusercontent.com/75513952/142361980-722dc284-2fc0-40b4-aafa-bf2faa33000a.png" width="800" height="70">

<p>PPO uses a on-policy update and clips the gradient descent step so learning is improved. The PPO objective function takes the minimum value between the original value and the clipped value. Positive advantage function means good action and negative advantage means bad action.</p>


### Resources Used 

- <https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html>
- <https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo>    
- <https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py>   
- <https://minecraft-archive.fandom.com/wiki/Blocks>    
- <https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f#:~:text=On%2DPolicy%20learning%20algorithms%20are,already%20using%20for%20action%20selection.>
- <https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-a-policy-based-reinforcement-learning-algorithm-3cf126a7562d>
