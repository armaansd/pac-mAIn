# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo XML docs: https://docs.ray.io/en/latest/rllib.html

# pacmAIn.py 
#
# model learned with PPO 
# Map is a 28 x 30 maze
# 52 diamonds the agent needs to collect

# To-do 
# PPO Currently doesnt work well. Maybe with more runs

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import torch
from torch import nn
import torch.nn.functional as F

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class MyModel(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self.obs_size = 25

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1) # 32, 5, 5 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32, 5, 5 
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32, 5, 5 

        self.policy_layer = nn.Linear(32*self.obs_size*self.obs_size, 3)
        self.value_layer = nn.Linear(32*self.obs_size*self.obs_size, 1)

        self.value = None
    
    def forward(self, input_dict, state, seq_lens):
        x  = input_dict['obs'] # BATCH, 2, 5, 5

        x = F.relu(self.conv1(x)) # BATCH 32, 5, 5 
        x = F.relu(self.conv2(x)) # BATCH 32, 5, 5 
        x = F.relu(self.conv3(x)) # BATCH 32, 5, 5 

        x = x.flatten(start_dim=1) # BATCH, 800

        policy = self.policy_layer(x) # BATCH, 3
        self.value = self.value_layer(x) # BATCH, 1

        return policy, state
    
    def value_function(self):
        return self.value.squeeze(1) # BATCH


class Pacman(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.runs = 0
        self.size = 20
        self.obs_size = 25   # Set back to 100 later
        self.max_episode_steps = 500
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        #self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = Box(-2, 1, shape=(2, self.obs_size, self.obs_size), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # Pacman Parameters
        self.obs = None
        self.episode_step = 0  # Steps in the episode
        self.steps_taken = 0   # Steps agent took to find the solution
        self.episode_return = 0
        self.episode_number = 0
        self.diamonds_collected = 0
        self.returns = []
        self.steps = []
        self.diamonds = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.steps_taken)

        self.diamonds.append(self.diamonds_collected)

        self.episode_return = 0
        self.episode_step = 0
        self.steps_taken = 0
        self.diamonds_collected = 0
        self.episode_number+= 1

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()
            self.log_diamonds()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        command = self.action_dict[action]
        #print("action {}\n".format(command))

        self.agent_host.sendCommand(command)
        time.sleep(.2)
        self.episode_step += 1
        if(self.diamonds_collected != 52):
            self.steps_taken += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            r_value = r.getValue()
            reward += r_value
            if(r_value == 1):
                self.diamonds_collected += r_value

        # Checking if agent is near a zombie
        # Reward gets decremeneted if agent is touching zombie
        # Since there's no "RewardForTouchingEntity" we check this by finding the position of the agent and the position of the zombie
        if(self.is_near_zombie(world_state) == True):
            print("Too close to Zombie!!\n")
            reward -= 1   # Decrement reward
            self.agent_host.sendCommand("quit")
            print("Agent ran into a Zombie! Agent died :(")

        self.episode_return += reward

        print()
        print("Episode Step " + str(self.episode_step) + "  Actual Step: " + str(self.steps_taken))
        print("Rewards gained: {}".format(self.episode_return))
        print("Diamonds collected: {}".format(self.diamonds_collected))
        print()

        if(self.diamonds_collected == 52): # Quit when reaching 52 diamonds 
            print("Collected all diamonds!\n")
            print("Steps taken: {}\n".format(self.steps_taken))
            self.agent_host.sendCommand("quit")

        return self.obs, reward, done, dict()


    def get_mission_xml(self):
        walls = draw_outer_wall()
        inner_walls = draw_inner_wall()
        diamonds = drawDiamond()
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Pacman</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>13000</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size-1, -(self.size+2), self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='stone'/>".format(-self.size, self.size-1, -(self.size+2), self.size) + \
                                walls + \
                                inner_walls +\
                                diamonds + \
                               '''
                                <DrawBlock x='0'  y='1' z='-14' type='redstone_block' />
                                <DrawBlock x='-1'  y='1' z='-14' type='redstone_block' />
                                <DrawBlock x='0'  y='1' z='12' type='grass' />
                                <DrawBlock x='-1'  y='1' z='12' type='grass' />
                                <DrawEntity x='0.5'  y='2' z='11.5' type='Zombie' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>CS175Pacman</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="-13.5" pitch="25"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_pickaxe"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>  
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid> 
                            <ObservationFromNearbyEntities>
                                <Range name="itemAll" xrange="'''+ str(int(self.obs_size/2)) + '''" yrange='2' zrange="'''+ str(int(self.obs_size/2)) + '''" />                                
                            </ObservationFromNearbyEntities>
                            <RewardForCollectingItem>
                                <Item reward="1" type="diamond"/>
                            </RewardForCollectingItem>
                            <RewardForTouchingBlockType>
                                <Block reward="-10" type="cobblestone"/> 
                            </RewardForTouchingBlockType>
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''"/>
                            <MissionQuitCommands quitDescription="found_all_diamonds"/>
                            <RewardForMissionEnd rewardForDeath="-2">
                                <Reward description="encountered_zombie" reward="-1"/>
                                <Reward description="found_all_diamonds" reward="1"/>
                            </RewardForMissionEnd>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(1000, 700)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'Pacman' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
        """
        obs = np.zeros((2, self.obs_size, self.obs_size))
        allow_break_action = False

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)  

                # Get observation
                grid = observations['itemAll']
                obs = obs.flatten()
                for item in grid:
                    index = (int)(self.obs_size * self.obs_size/2) + (int)(item['x'] - grid[0]['x']) + (int)(item['z'] - grid[0]['z']) * self.obs_size
                    if(item['name'] == 'diamond'):
                        obs[index] = 1
                    if(item['name'] == 'Zombie'):
                        obs[index] = -1

                # We also enumerate the walls to be -2 so the agent knows that it's walking into walls

                grid2 = observations['floorAll']
                for i, x in enumerate(grid2):
                    if(x == 'cobblestone'):
                        obs[i] = -2

                # Rotate observation with orientation of agent
                obs = obs.reshape((2, self.obs_size, self.obs_size))
                yaw = observations['Yaw']
                if yaw >= 225 and yaw < 315:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw >= 315 or yaw < 45:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw >= 45 and yaw < 135:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                #obs = obs.flatten()
                
                break
        #print(obs)

        return obs

    def is_near_zombie(self, world_state):
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)  
            
            # Get observation
            grid = observations['itemAll']
            #print(grid)
            for item in grid:
                x_dist = 1000
                z_dist = 1000
                if(item['name'] == 'Zombie'):
                    # Handle negative coordinates
                    if(item['x'] >= 0 and grid[0]['x'] >= 0):
                        x_dist = item['x'] - grid[0]['x']
                    elif(item['x'] <= 0 and grid[0]['x'] <= 0):
                        x_dist = abs(item['x']) - abs(grid[0]['x'])
                    elif(item['x'] <= 0 and grid[0]['x'] >= 0):
                        x_dist = abs(item['x']) + grid[0]['x']
                    elif(item['x'] >= 0 and grid[0]['x'] <= 0):
                        x_dist = item['x'] + abs(grid[0]['x'])
                    
                    if(item['z'] >= 0 and grid[0]['z'] >= 0):
                        z_dist = item['z'] - grid[0]['z']
                    elif(item['z'] <= 0 and grid[0]['z'] <= 0):
                        z_dist = abs(item['x']) - abs(grid[0]['x'])
                    elif(item['z'] <= 0 and grid[0]['z'] >= 0):
                        z_dist = abs(item['z']) + grid[0]['z']
                    elif(item['z'] >= 0 and grid[0]['z'] <= 0):
                        z_dist = item['z'] + abs(grid[0]['z'])

                    x_dist = abs(x_dist)
                    z_dist = abs(z_dist)
                    
                    if( (x_dist < 2 and x_dist >= 0) and (z_dist < 2 and z_dist >= 0)  ):
                        print("X: {} Z: {}".format(x_dist, z_dist))
                        return True

        return False
        

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Pacman: Total Steps v. Returns')
        plt.ylabel('Returns')
        plt.xlabel('Steps')
        plt.savefig('pacman_returns.png')

        with open('pacman_returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value)) 

        self.runs = self.runs + 1
        print("Run number: {}".format(self.runs))
    
    # Graph of episode number vs the amount of reward collected
    # Used to compare how many rewards agent received per episode
    def log_diamonds(self):

        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.diamonds[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Pacman: Total Steps v. Diamonds')
        plt.ylabel('Diamonds Collected')
        plt.xlabel('Steps')
        plt.savefig('pacman_diamonds_collected.png')

        with open('pacman_diamonds_collected.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.diamonds[1:]):
                f.write("{}\t{}\n".format(step, value)) 

# Spawn diamonds around the maze
def drawDiamond():
    diamond = ""
    for x in [-12, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8, 10, 11]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(x,12)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(x,12)
    
    for x in [-12, -11, -9, -7, -5, -3, 2, 4, 6, 8, 10]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(x,-14)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(x,-14)
        
    for z in [-14, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(11,z)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(11,z)

    for z in [-13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(-12,z)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(-12,z)
    return diamond

# Draw the outer wall  28 X 31 maze
def draw_outer_wall():
    outer_wall = ""
    for x in range(-14, 14):
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x, 14)
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x, -16)
        outer_wall += "<DrawBlock x ='{}' y='1' z='{}' type='sandstone' />".format(x, 14)
        outer_wall += "<DrawBlock x ='{}' y='1' z='{}' type='sandstone' />".format(x, -16)

        for y in [3, 4]:
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(x, y, 14)
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(x, y, -16)

        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(x,14)
        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(x,-16)

    for z in range(-16, 15):
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(13, z)
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(-14, z)
        outer_wall += "<DrawBlock x ='{}' y='1' z='{}' type='sandstone' />".format(13, z)
        outer_wall += "<DrawBlock x ='{}' y='1' z='{}' type='sandstone' />".format(-14, z)

        for y in [3, 4]:
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(13, y, z)
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(-14, y, z)

        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(13,z)
        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(-14,z)

    return outer_wall

# Draw the inner walls
def draw_inner_wall():
    inner_wall = ""
    for x in range(-10, 10):
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x, 10)
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x, -12)
        inner_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='BLUE' />".format(x, 10)
        inner_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='BLUE' />".format(x, -12)
    for z in range(-12, 11):
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(9, z)
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(-10, z)
        inner_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='BLUE' />".format(9, z)
        inner_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='BLUE' />".format(-10, z)

    for x in range(-9, 9):
        for z in range(-11, 10):
            inner_wall += "<DrawBlock x ='{}' y='4' z='{}' type='stained_glass' colour='YELLOW' />".format(x, z)
            inner_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='YELLOW' />".format(x, z)
            inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='torch' />".format(x,z)

    return inner_wall


if __name__ == '__main__':
    ModelCatalog.register_custom_model('my_model', MyModel)
    ray.init()
    trainer = ppo.PPOTrainer(env=Pacman, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0,            # We aren't using parallelism
        'model': {
            'custom_model' : 'my_model',
            'custom_model_config' : {}
        }
    })

    i = 0
    while True:
        result = trainer.train()
        print(result)
        print("Iteration: {}\n".format(i))
        if i % 2 == 0:
            checkpoint_path = trainer.save()
            print("checkpoint saved")   
            print(checkpoint_path)
        i += 1

