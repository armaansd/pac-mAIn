--- 
layout: default
title: Final Report
---
This will be exchanged for final vid later
<H2 align=left>Video</H2>
<iframe width="560" height="315" src="https://www.youtube.com/embed/Zw_zDCuyh54" frameborder="0" allowfullscreen>
</iframe>


<H2 align = left>Project Summary</H2>

### Project background: 

<p>Suppose a Minecraft player wanted to collect items around the map and is aware that there are hostile mobs nearby. This scenario is similar to the setup of PacMan; a player collects items in a maze while evading enemies. Thus, we framed our project around this setup.</p>

### Goal: 
  
<p>The goal of our project is to train our agent to play a modified version of Pac-Man on Minecraft. We will develop our AI using the Malmo platform. The agent will be placed in a maze to explore. The pellets from the original game will be substituted with diamonds and the ghosts will be substituted by a Zombie. The agent's score will be based on how many diamonds they collect in a given episode. The agent will "win" if it is able to collect all diamonds.</p>
    
<p>Of course, this project would be trivial if the agent only needed to collect diamonds. The agent will also have to learn to avoid the zombie as it picks up diamonds. If it gets near a zombie it will receive a penalty score. If it gets touched by the zombie, the agent will "die" and received a larger penalty score. Thus, the agent will have to learn to maneuver itself with this in mind. Because the position of the zombie changes over time and the diamonds disappear when they are collected, ML and/or RL algorithms can help the agent make reasonable actions in an environment with dynamic entities. It will be interesting to see if our agent will develop strategies for collecting diamonds similar to how a human might do. 
</p>

<p>We will create the environment ourselves and train our agent using two different approaches. We will then evaluate our agent based on several metrics.</p>


<img src="https://user-images.githubusercontent.com/75513952/144721188-2a29f8d2-261c-4e87-98e4-140f43b1356b.png" width="1000" height="700">



### Environement Setup
<p>Comapared to the previous version of the map in the status report, we made some changes to the environment. Because the agent is enclosed by walls, one behavior it learned was to not move to minimize the amount of negative rewards that it receives from touching the wall. Thus, it learned to stand around instead of exploring the maze. To encourage the agent to explore, we added more walking space for the agent. This also allows it to maneuver around a Zombie if it learns to do so. </p>

### Environment
- Enclosed 21 x 21 Map
- 31 Diamonds
- Zombie spawned randomly in one of three locations on the map. 



##### Below is the layout of Map 1 
<img src="https://user-images.githubusercontent.com/75513952/144721262-77b532d9-a85a-4b08-8b9a-ee5a24c4e50a.png" width="700" height="500">

### Rewards
We defined the following rewards:

Positive rewards
- Collecting Diamond +1
- Collecting all diamonds +100

Negative Rewards
- Near Zombie -1
- Touched/damaged by Zombie -5
- Touching wall -10



<H2 align=left>Approach</H2>

<p>Our minecraft agent was developed on Malmo. We trained on two different reinforcement learning algorithms. We wanted to explore on-policy and off-policy algorithms, so we decicded to train one agent using Proximal Policy Optimization (on-policy) and train another using Q-Learning (off-policy).</p>


### Approach 1: PPO
<p>One of the algorithms we used is Proximal Policy Optimization or PPO for short. We used the pre-implemented version of the PPO algorithm trainer from RLlib.
PPO is a on-policy algorithm, meaning that it explores by sampling actions based on the latest version of its policy. Essentially our agent learns from the actions it took with its current policy and then updates its policy in small batches and in multiple training steps. PPO uses a on-policy update and clips the gradient descent step so learning is improved. Initially the actions the agent will perform will be based on it's initial conditions and training procedure, but should get less random as more training goes on. </p>

PPO uses the update function:

<img src="https://user-images.githubusercontent.com/75513952/145362880-df9c5a00-04a8-4de1-820c-b28acd98d030.png" width="500" height="50">

where r(θ) is the ratio of the current policy and the old policy

<img src="https://user-images.githubusercontent.com/75513952/145363405-8d41a536-864d-4d2d-9a68-feaf5c48e474.png" width="150" height="50">


#### Observation Space
<p>In our scenario, we used a 3 x 17 x 17 image shape for the observation. We utilized 3 channels: one each for diamond, zombie, and wall blocks.</p>
<p>To preserve spatial information, we defined a custom NN model with three convutional layers. </p>

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
# Where X and Z are the x,z coordinates of the entity and x and z are the x,z coordinates of the agent
```

<p>After updating the observation array, the distance between the agent and zombie are checked. If the agent and the zombie are within touching distance or has been attacked by the zombie, the agent is considered to be "dead" and the mission will end.</p>

### Approach 2: Q-Learning
<p>Q-Learning is an off-policy algorithm, meaning the updated policy is different from the behavior policy. Unlike an on-policy algorithm, off-policy algorithms learn the value of the optimal policy independently of the agent’s actions. It updates its q-table using the q-value estimate of the next state. We used a simple tabular approach to implement Q-Learning. </p>

Q-Learning uses the following equation to update its table where Q(S, A) is the expected value of performing action a in state s

<img src="https://user-images.githubusercontent.com/75513952/145320504-4b8fa938-7b7d-494e-a9d9-2ea53c108fcd.png" width="400" height="200">

- alpha: Learning rate 
- R: reward 
- gamma: discount factor 
- Q(S, A): old value
- Q(S', a'): Estimate of optimal future value

- We used the following arguments:
```python
agent_host.addOptionalFloatArgument('alpha','Learning rate of the Q-learning agent.', 0.1)
agent_host.addOptionalFloatArgument('epsilon','Exploration rate of the Q-learning agent.', 0.01)
agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 1.0)
```
- We used discrete actions and defined the action space as follows: 
```python
actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
```

#### Updating q-table 

We used the following code snippet to update the q-table. This is computed by adding the old q-value with an estimate shown above in the diagram.  

```python 
if self.training and self.prev_s is not None and self.prev_a is not None:
    old_q = self.q_table[self.prev_s][self.prev_a]
    self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * (current_r + self.gamma * max(self.q_table[current_s]) - old_q)
```

- Q-table being filled at the start
<img src="https://user-images.githubusercontent.com/75513952/145655718-523fe51d-6d58-4d93-98d8-0459fe9f6794.JPG" width="500" height="500">

- Q-table after agent explores the map
<img src="https://user-images.githubusercontent.com/75513952/145660663-e091b3cb-c2ee-4684-a166-0aa67bd70d77.JPG" width="500" height="500">


<H2 align=left>Evaluation</H2>

## PPO: Quantitative
We generated three graphs to visualize and analyze the agent's performance over time. We will evaluate the number of diamonds the agent is able to collect, the total rewards, and the number of steps it takes to reach the solution (collect all diamonds).

## PPO Map 1

<img src="https://user-images.githubusercontent.com/75513952/145304596-c8dda948-edbd-4c0b-b956-f034a6577d72.png" width="600" height="400">

<p>Map 1 is basic maze with one path that goes around the maze and is the baseline for our experiments.  </p>


### Number of diamonds collected: 
<img src="https://user-images.githubusercontent.com/75513952/144725745-ccee522c-9d18-45be-944f-c720c258fd6d.png" width="700" height="500">

At the start of training, the agent mostly performs random actions and collects only a few diamonds. As you can see on the graph, the number of diamonds the agent is able to collect increases over time.  

### Returns:
<img src="https://user-images.githubusercontent.com/75513952/144725748-c2ff28ba-ec1f-45ad-a989-a56317c25a6c.png" width="700" height="500">


At the start of training, the agent mostly performs random actions, resulting it in running into walls and the zombie. This resulted in the agent receiving large negative rewards. Over time, the number of rewards increase as the agent is able to collect more diamonds while avoiding the zombie. 


### Number of steps to find solution:

<img src="https://user-images.githubusercontent.com/75513952/144727105-9adcead7-d67e-4a30-b8cb-9483b7f009c3.png" width="700" height="500">

The following graph shows the number of steps it took to collect all diamonds. The total steps the agent is allowed to perform is 500 steps. Initially, the agent requires a lot of steps to reach the solution, with the maximum being 251 steps. Over time, the agent requires fewer steps to reach the solution, which the minimum number being 58 steps. The average amount of steps the agent performed was 101 steps, which is about 1/5th of the total steps the agent is allowed to perform. 

### Steps taken to reach solution
- Max: 251
- Min: 58
- Avg: 101

## PPO Map 2

<img src="https://user-images.githubusercontent.com/75513952/145305404-5a4ba770-1269-4d3b-bc87-a33e9aaf0069.png" width="600" height="400">

<p>We also tested our model on a map where the agent has multiple path options available. There are now 38 diamonds for the agent to collect.</p> 


### Number of diamonds collected: 
<img src="https://user-images.githubusercontent.com/75513952/145306472-cb10c674-b634-40cc-8380-c3ca1614577e.png" width="700" height="500">

We tested our model on map 2. Initially, the performance of the agent is unstable as it has not learned to explore the new path added to the map. Over time, the agent's performance becomes less unstable and is able to collect diamonds more consistently compared to at the start. 

### Returns:
<img src="https://user-images.githubusercontent.com/75513952/145305836-7a6981e5-6811-4843-9ac9-293c8289a5a2.png" width="700" height="500">

Likewise, the number of returns increases with time. 

### Number of steps to find solution:
<img src="https://user-images.githubusercontent.com/75513952/145306318-938d5c17-acda-4f5d-b642-3881dbebe621.png" width="700" height="500">

Similar to the analysis for Map 1, the number of steps the agent requires decreases with time, with the maximum number of steps being 493 and the minimum number of steps being 88. Since there is an extra path the agent has to explore, the agent has to perform more steps compared to Map 1. The total amount of steps allowed is still 500 steps. The average number of steps the agent required to reach the solution was 198, which is about 2/5ths the total allowed steps. 

### Steps taken to reach solution
- Max: 493
- Min: 88
- Avg: 198.5


## PPO: Qualitative
We will evaluate our agent's performance based on the expected behaviour: Avoiding walls and zombies. 

<img src="https://user-images.githubusercontent.com/75513952/144725748-c2ff28ba-ec1f-45ad-a989-a56317c25a6c.png" width="700" height="500">

Reviewing the total steps vs returns graph once more, we see a spike in the returns around the 10,000 to 20,000 steps mark. Although the rewards are still negative, this indicates that with time, the agent has learned to avoid walls. Since touching walls results in the agent receiving -10 reward points, it accounts for most of the negative returns initially. After the 30,000 steps mark, the returns are mostly positive, which indicates that the agent has succesfully learned to avoid touching the walls. 

When the agent gets touched by a zombie and dies, the current mission will end. The graph indicates that over time, the agent is able to receive higher returns. Higher returns indicate that the agent is able to survive long enough to accumulate more rewards. Thus, our agent accomplishes the expected behaviour. 

Below are videos demonstrating our agent performing the expected behavior.

#### Below is a video of an example run where a solution is found on Map 1

<iframe width="560" height="315" src="https://www.youtube.com/embed/4lxwPoD3CQI" frameborder="0" allowfullscreen>
</iframe>

#### Below is a video of an example run where a solution is found on Map 2

<iframe width="560" height="315" src="https://www.youtube.com/embed/co5hQgN6pi8" frameborder="0" allowfullscreen>
</iframe>

As shown in the demonstrations, the agent is adept at collecting diamonds. When the agent observes a nearby zombie, it will avoid it. 


## Evaluation: Q-Learning

### Quantitative 
<p>Compared to PPO, the agent trained with Q-Learning took longer to reach a solution (collect all diamonds).</p>

### Qualitative
<p>Compared to PPO, the agent trained with tabular Q-Learning was unable to effectively learn to avoid the zombie while collecting diamonds. Its performance was unstable and performed poorly in our new environment. A major limitation of tabular Q-learning is that it applies only to discrete action and state spaces. Due to the zombie being a moving entity, it made q-value entries involving player death inaccurate and resulted in inefficient learning. Thus, the agent trained under Q-Learning did not accomplish the expected behavior of avoiding zombies.</p>

<p>Nevertheless, the agent was able to collect all the diamonds on the map within 500 steps in a map with no zombie.</p>

#### Below is a video of an example run where a solution is found
<iframe width="560" height="315" src="https://www.youtube.com/embed/R2OfQQSMz48" frameborder="0" allowfullscreen>
</iframe>



<H2>Resources Used</H2>

- <https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html>
- <https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo>    
- <https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py>   
- <https://minecraft-archive.fandom.com/wiki/Blocks>    
- <https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c>
- <https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-a-policy-based-reinforcement-learning-algorithm-3cf126a7562d>
