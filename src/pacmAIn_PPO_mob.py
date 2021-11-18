# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo XML docs: https://docs.ray.io/en/latest/rllib.html

# pacmAIn.py 
#
# model learned with PPO 
# Map is a 28 x 30 maze
# 52 diamonds the agent needs to collect

# Currently no zombies 
# Status: Testing item gathering only for now

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

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

# Spawn diamonds around the maze
def drawDiamond():
    diamond = ""
    for x in [-13, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8, 10, 12]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(x,13)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(x,13)
    
    for x in [-13, -11, -9, -7, -5, -3, 2, 4, 6, 8, 10]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(x,-15)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(x,-15)
        
    for z in [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(12,z)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(12,z)

    for z in [-13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11]:
        diamond += "<DrawItem x ='{}' y='6' z='{}' type='diamond' />".format(-13,z)
        diamond += "<DrawBlock x ='{}' y='1' z='{}' type='stained_glass' colour='BLUE' />".format(-13,z)
    return diamond

# Draw the outer wall  28 X 31 maze
def draw_outer_wall():
    outer_wall = ""
    for x in range(-14, 14):
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,14)
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,-16)
        outer_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='WHITE' />".format(x,14)
        outer_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='WHITE' />".format(x,-16)
        outer_wall += "<DrawBlock x ='{}' y='4' z='{}' type='stained_glass' colour='WHITE' />".format(x,14)
        outer_wall += "<DrawBlock x ='{}' y='4' z='{}' type='stained_glass' colour='WHITE' />".format(x,-16)

        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(x,14)
        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(x,-16)
    for z in range(-16, 15):
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(13,z)
        outer_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(-14,z)
        outer_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='WHITE' />".format(13,z)
        outer_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='WHITE' />".format(-14,z)
        outer_wall += "<DrawBlock x ='{}' y='4' z='{}' type='stained_glass' colour='WHITE' />".format(13,z)
        outer_wall += "<DrawBlock x ='{}' y='4' z='{}' type='stained_glass' colour='WHITE' />".format(-14,z)

        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(13,z)
        outer_wall += "<DrawBlock x ='{}' y='5' z='{}' type='torch' />".format(-14,z)
    return outer_wall

# Draw the inner walls
def draw_inner_wall():
    inner_wall = ""
    for x in range(-12, 12):
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,12)
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,-14)
    for z in range(-14, 13):
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(11,z)
        inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(-12,z)

    for x in range(-11, 11):
        for z in range(-13, 12):
            inner_wall += "<DrawBlock x ='{}' y='4' z='{}' type='stained_glass' colour='YELLOW' />".format(x, z)
            inner_wall += "<DrawBlock x ='{}' y='3' z='{}' type='stained_glass' colour='YELLOW' />".format(x, z)
            inner_wall += "<DrawBlock x ='{}' y='2' z='{}' type='torch' />".format(x,z)

    return inner_wall

# Original pacman maze might be too complex to use
#def draw_maze():
#    # Half the maze is basically a mirror of itself
#    maze = draw_outer_wall()
#    for x in range(-13, 13):
#        if(x == 12 or x == -13):
#            for z in [5, 1, -1, -5, -10, -11]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 11 or x == -12):
#            for z in [12, 11, 10, 8, 7, 5, 1, -1, -5, -7, -8, -10, -11, -13, -14]:
#                    maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 10 or x == -11):
#            for z in [12, 11, 10, 8, 7, 5, 1, -1, -5, -7, -8, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 9 or x == -10):
#            for z in [12, 11, 10, 8, 7, 5, 1, -1, -5, -7, -8, -9, -10, -11, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 8 or x == -9):
#            for z in [12, 11, 10, 8, 7, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -7, -8, -9, -10, -11, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 7 or x == -8):
#            for z in [-13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 6 or x == 5 or x == -7 or x == -6):
#            for z in [12, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -7, -8, -10, -11, -12, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 4 or x == -5):
#            for z in [12, 11, 10, 5, 4, -7, -8, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 3 or x == -4):
#            for z in [12, 11, 10, 8, 7, 5, 4, 2, 1, 0, -1, -2, -4, -5, -7, -8, -10, -11, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 2 or x == -3):
#            for z in [12, 11, 10, 8, 7, 5, 4, 2, -2, -4, -5, -7, -8, -10, -11, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 1 or x == -2):
#            for z in [8, 7, 2, -2, -4, -5, -10, -11]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z)
#        elif(x == 0 or x == -1):
#            for z in [13, 12, 11, 10, 8, 7, 6, 5, 4, -2, -4, -5, -6, -7, -8, -10, -11, -12, -13, -14]:
#                maze += "<DrawBlock x ='{}' y='2' z='{}' type='cobblestone' />".format(x,z) 
#    return maze    

class Pacman(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.runs = 0
        self.size = 20
        self.obs_size = 100
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
        self.observation_space = Box(-1, 1, shape=(2 * self.obs_size * self.obs_size, ), dtype=np.float32)

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
        self.allow_break_action = False
        self.episode_step = 0  # Steps in the episode
        self.steps_taken = 0   # Steps agent took to find the solution
        self.episode_return = 0
        self.episode_number = 0
        self.diamonds_collected = 0
        self.returns = []
        self.steps = []
        self.episode_step_arr = []
        self.episode_arr = []

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

        self.episode_step_arr.append(self.steps_taken)
        self.episode_arr.append(self.episode_number)

        self.episode_return = 0
        self.episode_step = 0
        self.steps_taken = 0
        self.diamonds_collected = 0
        self.episode_number+= 1

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()
            self.log_steps()
            self.log_rewards_per_episode()

        # Get Observation
        self.obs, self.allow_break_action = self.get_observation(world_state)

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
        self.obs, self.allow_break_action = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            r_value = r.getValue()
            reward += r_value
            if(r_value != -1):
                self.diamonds_collected += r_value

        self.episode_return += reward

        print()
        print("Episode Step " + str(self.episode_step) + "  Actual Step: " + str(self.steps_taken))
        print("Rewards gained: {}".format(self.episode_return))
        print()

        if(self.diamonds_collected == 52): # Quit when reaching 52 diamonds 
            print("Collected all diamonds!\n")
            print("Steps taken: {}\n".format(self.steps_taken))

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
                                <DrawBlock x='0'  y='1' z='-15' type='redstone_block' />
                                <DrawBlock x='-1'  y='1' z='-15' type='redstone_block' />
                                <DrawBlock x='0'  y='1' z='13' type='grass' />
                                <DrawBlock x='-1'  y='1' z='13' type='grass' />
                                <DrawEntity x='0.5'  y='2' z='12.5' type='Zombie' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>CS175Pacman</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="-14.5" pitch="25"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_pickaxe"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>   
                             <ObservationFromNearbyEntities>
                                <Range name="itemAll" xrange="'''+ str(self.obs_size) + '''" yrange='2' zrange="'''+ str(self.obs_size) + '''" />                                
                            </ObservationFromNearbyEntities>
                            <RewardForCollectingItem>
                                <Item reward="1" type="diamond"/>
                            </RewardForCollectingItem>
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''"/>
                            <MissionQuitCommands quitDescription="encountered_zombie"/>
                            <RewardForMissionEnd>
                                <Reward description="encountered_zombie" reward="-1"/>
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
        my_mission.requestVideo(1200, 900)
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
        obs = np.zeros((2 * self.obs_size * self.obs_size, ) )
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
                print(grid)
                for item in grid:
                    index = (int)(self.obs_size * self.obs_size/2) + (int)(item['x'] - grid[0]['x']) + (int)(item['z'] - grid[0]['z']) * self.obs_size
                    if(item['name'] == 'diamond'):
                        obs[index] = 1
                    if(item['name'] == 'Zombie'):
                        obs[index] = -1
                        print("Zombie near!")
                        
                        x_dist = item['x'] - grid[0]['x']
                        z_dist = item['z'] - grid[0]['z']

                        if(x_dist <= 1 and x_dist >= -1 and z_dist <= 1 and z_dist >= -1):
                            self.agent_host.sendCommand("quit")
                    

                print(obs)

                # Rotate observation with orientation of agent
                obs = obs.reshape((2, self.obs_size, self.obs_size))
                yaw = observations['Yaw']
                if yaw >= 225 and yaw < 315:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw >= 315 or yaw < 45:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw >= 45 and yaw < 135:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                obs = obs.flatten()
                
                break

        return obs, allow_break_action

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
    
    # Graph of episode number vs the amount of steps
    # Used to compare how many steps agent took per episode
    def log_steps(self):

        box = np.ones(self.log_frequency) / self.log_frequency
        episode_smooth = np.convolve(self.episode_step_arr[1:], box, mode='same')
        plt.clf()
        plt.plot(self.episode_arr[1:], episode_smooth)
        plt.title('Pacman: Episodes v. Steps')
        plt.ylabel('Steps Taken')
        plt.xlabel('Episode No.')
        plt.savefig('pacman_steps.png')

        with open('pacman_steps.txt', 'w') as f:
            for step, value in zip(self.episode_arr[1:], self.episode_step_arr[1:]):
                f.write("{}\t{}\n".format(step, value)) 
    
    # Graph of episode number vs the amount of reward collected
    # Used to compare how many rewards agent received per episode
    def log_rewards_per_episode(self):

        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.episode_arr[1:], returns_smooth)
        plt.title('Pacman: Episodes v. Reward')
        plt.ylabel('Returns')
        plt.xlabel('Episode No.')
        plt.savefig('pacman_rewards_per_episode.png')

        with open('pacman_rewards_per_episode.txt', 'w') as f:
            for step, value in zip(self.episode_arr[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value)) 

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=Pacman, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
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

