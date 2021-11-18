--- 
layout: default
title: Status
---
<iframe width="560" height="315" src="https://www.youtube.com/watch?v=GXTNISa5NKk" frameborder="0" allowfullscreen></iframe>
</iframe>
      
# Project Summary
<p>The goal of our project is to train our agent to play a modified Mincraft recreation of the game Pac-Man. The goal of our agent is to get as many points as possible before time runs out (500 steps). The agent will be placed in a 28 x 31 enclosed maze and will have to traverse the map in order to pick up diamonds located around the maze. The agent's score will be based on how many diamonds they collect in a given episode. There will be a total of 52 diamonds for the agent to collect. We will develop our AI using Malmo. </p>

<p>In this maze will also include a zombie, replacing the ghost in the original Pac-Man game. The agent will have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie and touches it, it will receive a penalty score.</p>

<p>The goal of this project is to implement the environment ourselves and implement and compare more than one algorithm. </p>

<img src="https://user-images.githubusercontent.com/75513952/142336340-20a53401-44f9-48f4-a5fd-9d6d77205444.png" width="900" height="700">

### Environement Setup
- 28 x 31 Map
- 52 Diamonds

### Rewards
- Diamond +1
- Touching Zombie -1


# Approach
## Algorithm Used: PPO
<p>One of the algorithms we are exploring is Proximal Policy Optimization or PPO for short. We used RLlib's implementation of a PPO trainer.
PPO is a on-policy algorithm, meaning that it explores by sampling actions based on its latest version of its stochastic policy. Essentially our agent learns from actions that it took using its current optimized policy and then updates its optimized policy. Initially the actions the agent will perform will be based on it's initial conditions and training procedure, but should get less random as more training goes on. Eventually this causes the agent to exploit already discovered rewards. </p>

<p>In our scenario, an 2 x 5 x 5 observation grid for entities near the agent will be generated. Diamonds will be enumerated with the value of 1. Zombies will be enumerated with a value of -1. Since diamonds will be removed from the map whenever the agent picks it up, an observation grid for entities is needed instead of for blocks around the agent. Adjusting the index of the item according to the agent will help update the agent's obervation, reward, and states to update its next policy.</p>
  
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


# Evaluation
### Quantitative:
<p> Because the goal of Pac-Man is to gather points, our current evaluation is based on how many points our agent obtained per episode and how many steps it took per episode. The agent receives plus 1 reward for collecting diamonds and negative 1 reward for touching a zombie. Each episode length is at most 500 steps. Since we only tested our agent's ability to gather diamonds, our goal was to evaluate how it would perform without an enemy in the way. Thus the following graphs represent metrics without encountering zombies. </p>

<p>Our first metric evaluation is total steps versus the reward the agent obtained. As we can see, initially the agent is not able to gather all diamonds on the map. However, as more training goes on, the amount of reward the agent is able to obtain increases. After 15,000 steps the agent is able to obtain all or most diamonds on the map.</p>

- Discrete movement
<img src="https://user-images.githubusercontent.com/75513952/142354740-e98327f3-f642-49d9-a9ba-f146a04f9415.png" width="500" height="300">

<p>Our second metric evaluation is similar to the previous one. We compared the the number of episodes versus the reward the agent obtained. Similar to the previous metrics, the amount of reward increases as more episodes occur. We can see after 30 episodes, the agent starts to consistently get most of the diamonds on the map. </p>

- Discrete Movement
<img src="https://user-images.githubusercontent.com/75513952/142354749-c8be969b-3c21-4112-ad19-530c700fc8fe.png" width="500" height="300">

<p>Our third metric is episode versus steps. This means how many steps the agent took per episode. Ideally the graph should start having a negative slope as more training occurs. This is because as the agent gets better at collecting diamonds, it should require less steps per episode. Initially, the agent takes 500 steps per episode, which is the maximum. However around 30 episodes, it starts to take less steps.</p>

- Discrete Movement
<img src="https://user-images.githubusercontent.com/75513952/142354766-c26b8869-5cb0-458a-8687-d68c33e14d96.png" width="500" height="300">

### Qualitative:
<p>One thing to note, there was a bug during our mission runs. Sometimes one or two diamonds don't get spawned properly so there were less than 52 diamonds for the agent to collect. Thus one method we evaluated our agent was to see how consistent the agent was at performing its task. As we can see from the graphs above, the agent was able to consistently gather 45 to 52 diamonds per run after 15000 steps. This could be improved with more runs.</p>

<p> We also compared graphs for continuous movement and found that PPO with continuous movement did not result in much improvement compared to discrete movement. Unlike discrete movement, training with continuous movement resulted in more fluctuations with the amount of rewards our agent was able to obtain. Our agent under continuous movement was unable to collect most or all the diamonds on the map. </p>

- Continuous Movement
<img src="https://user-images.githubusercontent.com/75513952/142358028-684667f9-d2a5-4bc7-a812-da3ca2273885.png" width="500" height="300">
<img src="https://user-images.githubusercontent.com/75513952/142358057-b3b707a7-df9b-4e5f-bf39-4c8345f7771e.png" width="500" height="300">
<img src="https://user-images.githubusercontent.com/75513952/142358084-1dca03ad-ec68-44cb-88c0-a01c769ea037.png" width="500" height="300">


# Remaining Goals and Challenges

<p> 
Currently our prototype is limited as we were testing PPO on a map without mobs. We wanted to get a measure of how successful our agent can collect diamonds. This was our baseline. Our goal will then be to include at least one zombie or enemy agent in which our agent will need to avoid. Since the zombie or enemy agent will be moving, it will not be static, so this may pose a challenge for our learning algorithms. We could improve on our agent's observation and action policy in order to solve this. 
</p>
  
<p>
Since PPO is an on-policy algorithm, we would like to try an off-policy algorithm, such as Q-learning. Our remaining goal is to implement our Pac-man AI using tabular Q-Learning. We will then compare it with our agent learned on PPO. 
</p>

### Resources Used 

- <https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html>
- <https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo>    
- <https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py>   
- <https://minecraft-archive.fandom.com/wiki/Blocks>    
- <https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f#:~:text=On%2DPolicy%20learning%20algorithms%20are,already%20using%20for%20action%20selection.>
- <https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-a-policy-based-reinforcement-learning-algorithm-3cf126a7562d>

