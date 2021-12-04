--- 
layout: default
title: Final Report
---
This will be exchanged for final vid later
<iframe width="560" height="315" src="https://www.youtube.com/embed/GXTNISa5NKk" frameborder="0" allowfullscreen>
</iframe>

<img src="https://user-images.githubusercontent.com/75513952/144721188-2a29f8d2-261c-4e87-98e4-140f43b1356b.png" width="1000" height="700">


# Pac-mAIn Project Summary
<p>The goal of our project is to train an agent to play a modified version of Pac-Man on Minecraft. The pellets from the original game will be substituted with diamonds and the ghosts will be substituted by a Zombie or enemy agent. The agent's score will be based on how many diamonds they collect in a given episode. The goal of our agent is to obtain the highest score for the given map, which is 31 diamonds. The agent will also have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie it will receive a penalty score. If it gets touched by the zombie, agent will "die" and received a penalty score. We will develop our AI using the Malmo platform.</p>

<p>The goal of this project is to create the environment ourselves and implement and compare more than one algorithm. We will evaluate our agent based on several metrics.</p>


### Environement Setup
<p>We made some changes to the environment since the proposal. Because the agent is enclosed with walls, one behavior it learned was to not move to minimize the amount of negative rewards that it receives from touching the wall. Thus, it learned to stand around instead of exploring the maze. To encourage the agent to explore, we added more walking space for the agent. This also allows it to maneuver around a Zombie if it learns to do so. </p>

- 21 x 21 Map
- 31 Diamonds
- Zombie spawned in 3 random locations on the map


<img src="https://user-images.githubusercontent.com/75513952/144721262-77b532d9-a85a-4b08-8b9a-ee5a24c4e50a.png" width="700" height="500">

### Rewards
We defined the following rewards:
- Collecting Diamond +1
- Near Zombie -1
- Touched by Zombie -5
- Touching wall -10
- Collecting all diamonds +100

# Approach
## Algorithm Used: PPO
<p>One of the algorithms we used is Proximal Policy Optimization or PPO for short. We used RLlib's implementation of a PPO trainer.
PPO is a on-policy algorithm, meaning that it explores by sampling actions based on its latest version of its policy. Essentially our agent learns from the observations and reward states with its current policy and then updates its policy in small batches in multiple training steps. Initially the actions the agent will perform will be based on it's initial conditions and training procedure, but should get less random as more training goes on. </p>

### Observation Space
<p>In our scenario, we used a 3 x 17 x 17 image shape for the observation. We utilized 3 channels: one each for diamond, zombie, and wall blocks. To preserve spacial information, we defined a custom NN model with three convutional layers. </p>

```python
class MyModel(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self.obs_size = 17

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # 32, self.obs_size, self.obs_size
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32, self.obs_size, self.obs_size
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32, self.obs_size, self.obs_size

        self.policy_layer = nn.Linear(32*self.obs_size*self.obs_size, 4) # input is flattened, action size 4
        self.value_layer = nn.Linear(32*self.obs_size*self.obs_size, 1)

        self.value = None
    
    def forward(self, input_dict, state, seq_lens):
        x  = input_dict['obs'] # BATCH, 3, self.obs_size, self.obs_size

        x = F.relu(self.conv1(x)) # BATCH, 32, self.obs_size, self.obs_size
        x = F.relu(self.conv2(x))  # BATCH, 32, self.obs_size, self.obs_size
        x = F.relu(self.conv3(x)) # BATCH, 32, self.obs_size, self.obs_size
        
        x = x.flatten(start_dim=1) # Flattened

        policy = self.policy_layer(x) 
        self.value = self.value_layer(x) 

        return policy, state
    
    def value_function(self):
        return self.value.squeeze(1) 
```
  
We used discrete actions and defined the action space for PPO as follows:

### Action Space
``` python 
self.action_dict = {
    0: 'move 1', 
    1: 'turn 1',  
    2: 'turn -1', 
    3: 'move 1.5', 
}
```

## Diagram of PPO architecture from RLlib's algorithm website 
<img src="https://user-images.githubusercontent.com/75513952/142348893-9389ccb9-e4f3-40da-83f1-b252248ae35c.png" width="800" height="300">

## PPO defines a probability ratio between its new policy and old policy
```python
r(θ) = π<sub>θ</sub>(a given s) / π<sub>θold</sub>(a given s)
```

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
