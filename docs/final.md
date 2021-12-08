--- 
layout: default
title: Final Report
---
This will be exchanged for final vid later
<H2 align=left>Video</H2>
<iframe width="560" height="315" src="https://www.youtube.com/embed/Zw_zDCuyh54" frameborder="0" allowfullscreen>
</iframe>


<img src="https://user-images.githubusercontent.com/75513952/144721188-2a29f8d2-261c-4e87-98e4-140f43b1356b.png" width="1000" height="700">

<H2 align = left>Pac-mAIn Project Summary</H2>
<p>The goal of our project is to train an agent to play a modified version of Pac-Man on Minecraft. The pellets from the original game will be substituted with diamonds and the ghosts will be substituted by a Zombie or enemy agent. The agent's score will be based on how many diamonds they collect in a given episode. The goal of our agent is to obtain the highest score for the given map, which is 31 diamonds. The agent will also have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie it will receive a penalty score. If it gets touched by the zombie, agent will "die" and received a penalty score. We will develop our AI using the Malmo platform.</p>

<p>The goal of this project is to create the environment ourselves and implement and compare more than one algorithm. We will evaluate our agent based on several metrics.</p>

#### Environement Setup
<p>We made some changes to the environment since the proposal. Because the agent is enclosed with walls, one behavior it learned was to not move to minimize the amount of negative rewards that it receives from touching the wall. Thus, it learned to stand around instead of exploring the maze. To encourage the agent to explore, we added more walking space for the agent. This also allows it to maneuver around a Zombie if it learns to do so. </p>

#### Environment
- Enclosed 21 x 21 Map
- 31 Diamonds
- Zombie spawned in 3 random locations on the map


<img src="https://user-images.githubusercontent.com/75513952/144721262-77b532d9-a85a-4b08-8b9a-ee5a24c4e50a.png" width="700" height="500">

#### Rewards
We defined the following rewards:
- Collecting Diamond +1
- Near Zombie -1
- Touched by Zombie -5
- Touching wall -10
- Collecting all diamonds +100


<H2 align=left>Approach</H2>

#### Algorithm Used: PPO
<p>One of the algorithms we used is Proximal Policy Optimization or PPO for short. We used RLlib's implementation of a PPO trainer.
PPO is a on-policy algorithm, meaning that it explores by sampling actions based on its latest version of its policy. Essentially our agent learns from the observations and reward states with its current policy and then updates its policy in small batches in multiple training steps. Initially the actions the agent will perform will be based on it's initial conditions and training procedure, but should get less random as more training goes on. </p>

#### Diagram of PPO architecture from RLlib's algorithm website 
<img src="https://user-images.githubusercontent.com/75513952/142348893-9389ccb9-e4f3-40da-83f1-b252248ae35c.png" width="800" height="300">

#### Observation Space
<p>In our scenario, we used a 3 x 17 x 17 image shape for the observation. We utilized 3 channels: one each for diamond, zombie, and wall blocks. To preserve spatial information, we defined a custom NN model with three convutional layers. </p>

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

#### Action Space
``` python 
self.action_dict = {
    0: 'move 1', 
    1: 'turn 1',  
    2: 'turn -1', 
    3: 'move 1.5', 
}
```
#### Adjusting Observations
<p>Because our agent and the zombie are moving entities, their positions on the grid will change as they move around on the map. Diamonds will also disappear from the map as they are collected by the agent. Thus, we defined our observations with ObservationFromNearbyEntities and ObservationFromGrid. To adjust the observation grid in relation to the agent, we used the following equation. </p>

```python
index = math.floor((self.obs_size**2)/2) + math.floor(X-x) + math.floor(Z-z) * self.obs_size
# Where X and Z are the x and z coordinates of the entity and x and z are the x and z coordinates of the agent
```

<H2 align=left>Evaluation</H2>

#### PPO Map 1

![2021-12-04_10 42 10](https://user-images.githubusercontent.com/75513952/145304596-c8dda948-edbd-4c0b-b956-f034a6577d72.png)


<img src="https://user-images.githubusercontent.com/75513952/144725745-ccee522c-9d18-45be-944f-c720c258fd6d.png" width="700" height="500">

<img src="https://user-images.githubusercontent.com/75513952/144725748-c2ff28ba-ec1f-45ad-a989-a56317c25a6c.png" width="700" height="500">

##### The following graph shows the steps for the episodes where the agent was able to collect all the diamonds 

<img src="https://user-images.githubusercontent.com/75513952/144727105-9adcead7-d67e-4a30-b8cb-9483b7f009c3.png" width="700" height="500">


### Steps taken to reach solution
- Max: 251
- Min: 58
- Avg: 101

#### PPO Map 2

<img src="https://user-images.githubusercontent.com/75513952/145305404-5a4ba770-1269-4d3b-bc87-a33e9aaf0069.png" width="500" height="400">


<p>We also tested our model on a map where the agent has multiple path options available. There are now 38 diamonds for the agent to collect.</p> 

<img src="https://user-images.githubusercontent.com/75513952/145306472-cb10c674-b634-40cc-8380-c3ca1614577e.png" width="700" height="500">

<img src="https://user-images.githubusercontent.com/75513952/145305836-7a6981e5-6811-4843-9ac9-293c8289a5a2.png" width="700" height="500">


##### The following graph shows the steps for the episodes where the agent was able to collect all the diamonds 

<img src="https://user-images.githubusercontent.com/75513952/145306318-938d5c17-acda-4f5d-b642-3881dbebe621.png" width="700" height="500">


### Steps taken to reach solution
- Max: 493
- Min: 88
- Avg: 198.5


<H2>Resources Used</H2>

- <https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html>
- <https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo>    
- <https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py>   
- <https://minecraft-archive.fandom.com/wiki/Blocks>    
- <https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f#:~:text=On%2DPolicy%20learning%20algorithms%20are,already%20using%20for%20action%20selection.>
- <https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-a-policy-based-reinforcement-learning-algorithm-3cf126a7562d>
