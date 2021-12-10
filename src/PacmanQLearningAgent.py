from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import random
import malmoutils
import matplotlib.pyplot as plt


if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

save_images = False
if save_images:        
    from PIL import Image
    
malmoutils.fix_print()
size = 20
obs_size = 17   
max_episode_steps = 500

value = 10

# Spawn diamonds around the maze
def drawDiamond():
    diamond = ""
    for x in [-8, -6, -4, -2, 2, 4, 6, 8]:
        diamond += "<DrawItem x ='{}' y='150' z='{}' type='diamond' />".format(x+value,8+value)
        diamond += "<DrawBlock x ='{}' y='45' z='{}' type='stained_glass' colour='BLUE' />".format(x+value,8+value)
    
    for x in [-8, -6, -4, -2, 0, 2, 4, 6, 8]:
        diamond += "<DrawItem x ='{}' y='150' z='{}' type='diamond' />".format(x+value,-8+value)
        diamond += "<DrawBlock x ='{}' y='45' z='{}' type='stained_glass' colour='BLUE' />".format(x+value,-8+value)
        
    for z in [-6, -4, -2, 0, 2, 4, 6]:
        diamond += "<DrawItem x ='{}' y='150' z='{}' type='diamond' />".format(8+value,z+value)
        diamond += "<DrawBlock x ='{}' y='45' z='{}' type='stained_glass' colour='BLUE' />".format(8+value,z+value)

    for z in [-6, -4, -2, 0, 2, 4, 6]:
        diamond += "<DrawItem x ='{}' y='150' z='{}' type='diamond' />".format(-8+value,z+value)
        diamond += "<DrawBlock x ='{}' y='45' z='{}' type='stained_glass' colour='BLUE' />".format(-8+value,z+value)
    return diamond

# Draw the outer wall  28 X 31 maze
def draw_outer_wall():
    outer_wall = ""
    for x in range(-10, 11):
        outer_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(x+value, 10+value)
        outer_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(x+value, -10+value)
        outer_wall += "<DrawBlock x ='{}' y='45' z='{}' type='sandstone' />".format(x+value, 10+value)
        outer_wall += "<DrawBlock x ='{}' y='45' z='{}' type='sandstone' />".format(x+value, -10+value)

        for y in [47, 48]:
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(x+value, y, 10+value)
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(x+value, y, -10+value)

        outer_wall += "<DrawBlock x ='{}' y='49' z='{}' type='torch' />".format(x+value,10+value)
        outer_wall += "<DrawBlock x ='{}' y='49' z='{}' type='torch' />".format(x+value,-10+value)

    for z in range(-10, 11):
        outer_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(10+value, z+value)
        outer_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(-10+value, z+value)
        outer_wall += "<DrawBlock x ='{}' y='45' z='{}' type='sandstone' />".format(10+value, z+value)
        outer_wall += "<DrawBlock x ='{}' y='45' z='{}' type='sandstone' />".format(-10+value, z+value)

        for y in [47, 48]:
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(10+value, y, z+value)
            outer_wall += "<DrawBlock x ='{}' y='{}' z='{}' type='stained_glass' colour='WHITE' />".format(-10+value, y, z+value)

        outer_wall += "<DrawBlock x ='{}' y='49' z='{}' type='torch' />".format(10+value,z+value)
        outer_wall += "<DrawBlock x ='{}' y='49' z='{}' type='torch' />".format(-10+value,z+value)
    


    return outer_wall

# Draw the inner walls
def draw_inner_wall():
    inner_wall = ""
    for x in range(-6, 7):
        inner_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(x+value,6+value)
        inner_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(x+value,-6+value)
        inner_wall += "<DrawBlock x ='{}' y='47' z='{}' type='stained_glass' colour='BLUE' />".format(x+value, 6+value)
        inner_wall += "<DrawBlock x ='{}' y='47' z='{}' type='stained_glass' colour='BLUE' />".format(x+value, -6+value)
        inner_wall += "<DrawBlock x ='{}' y='48' z='{}' type='torch' />".format(x+value,6+value)
        inner_wall += "<DrawBlock x ='{}' y='48' z='{}' type='torch' />".format(x+value,-6+value)
        
    for z in range(-6, 7):
        inner_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(6+value,z+value)
        inner_wall += "<DrawBlock x ='{}' y='46' z='{}' type='cobblestone' />".format(-6+value,z+value)
        inner_wall += "<DrawBlock x ='{}' y='47' z='{}' type='stained_glass' colour='BLUE' />".format(6+value, z+value)
        inner_wall += "<DrawBlock x ='{}' y='47' z='{}' type='stained_glass' colour='BLUE' />".format(-6+value, z+value)
        inner_wall += "<DrawBlock x ='{}' y='48' z='{}' type='torch' />".format(6+value, z+value)
        inner_wall += "<DrawBlock x ='{}' y='48' z='{}' type='torch' />".format(-6+value,z+value)

    for x in range(-5, 6):
        for z in range(-5, 6):
            inner_wall += "<DrawBlock x ='{}' y='46' z='{}' type='torch' />".format(x+value,z+value)
            inner_wall += "<DrawBlock x ='{}' y='47' z='{}' type='stained_glass' colour='YELLOW' />".format(x+value, z+value)
            inner_wall += "<DrawBlock x ='{}' y='48' z='{}' type='stained_glass' colour='YELLOW' />".format(x+value, z+value)
    

    return inner_wall

# Randomly draws the zombie in 1 of 3 locations
def draw_zombie():
    num = random.randint(0, 2)

    zombie_xml = ""
    positions = [(-8.5, 0.5), (0.5, -8.5), (8.5, 0.5)]
    zombie_xml += "<DrawEntity x='{}'  y='46' z='{}' type='Zombie' />".format(positions[num][0], positions[num][1])
    #zombie_xml += "<DrawEntity x='{}'  y='2' z='{}' type='Zombie' />".format(-8.5, 8.5)
    
    return zombie_xml

def is_near_entity(entity_x, entity_z, agent_x, agent_z):  
    # args: entity X and Z position and agent's X and Z positions
    
    #print("CHECKING")
    touched = False
    dead = False
    
    x_dist = 1000  # Arbitrary large numbers
    z_dist = 1000  # Arbitrary large numbers

    # Handle negative coordinates
    # Getting distance between zombie item and agent (grid[0])
    if(entity_x >= 0 and agent_x >= 0):
        x_dist = entity_x - agent_x
    elif(entity_x <= 0 and agent_x <= 0):
        x_dist = abs(entity_x) - abs(agent_x)
    elif(entity_x <= 0 and agent_x >= 0):
        x_dist = abs(entity_x) + agent_x
    elif(entity_x >= 0 and agent_x <= 0):
        x_dist = entity_x + abs(agent_x)
    
    if(entity_z >= 0 and agent_z >= 0):
        z_dist = entity_z - agent_z
    elif(entity_z <= 0 and agent_z <= 0):
        z_dist = abs(entity_z) - abs(agent_z)
    elif(entity_z <= 0 and agent_z >= 0):
        z_dist = abs(entity_z) + agent_z
    elif(entity_z >= 0 and agent_z <= 0):
        z_dist = entity_z + abs(agent_z)

    x_dist = abs(x_dist)
    z_dist = abs(z_dist)

    if( (x_dist <= 1.5 and x_dist >= 0) and (z_dist <= 1.5 and z_dist >= 0)):
        touched = True
    
    # Considering "Close" to be less than 2 blocks away from the agent. The agent and zombie are practically touching
    if( (x_dist <= 3.0 and x_dist >= 0) and (z_dist <= 2.0 and z_dist >= 0) or (x_dist <= 2.0 and x_dist >= 0) and (z_dist <= 3.0 and z_dist >= 0) ):
        print("X: {} Z: {}".format(x_dist, z_dist))
        dead = True

    return (touched, dead)

def get_mission_xml():
    walls = draw_outer_wall()
    inner_walls = draw_inner_wall()
    diamonds = drawDiamond()
    zombie = ""

    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <About>
                    <Summary>Pacman</Summary>
                </About>
                <ModSettings>
                    <MsPerTick>50</MsPerTick>
                </ModSettings>
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
                            "<DrawCuboid x1='{}' x2='{}' y1='46' y2='50' z1='{}' z2='{}' type='air'/>".format(-size+value, size-1+value, -(size+2)+value, size+value) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='45' y2='45' z1='{}' z2='{}' type='stone'/>".format(-size+value, size-1+value, -(size+2)+value, size+value) + \
                            walls + \
                            inner_walls +\
                            diamonds + \
                            zombie + \
                            "<DrawBlock x='{}'  y='45' z='{}' type='redstone_block' />".format(0+value, 8+value) + \
                            "<DrawBlock x='{}'  y='45' z='{}' type='grass' />".format(0+value, -8+value) +\
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>CS175Pacman</Name>
                    <AgentStart> ''' +\
                        "<Placement x='{}' y='46' z='{}' yaw = '180' pitch='25'/>".format(0.5+value, 8.5+value) + \
                        '''
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_pickaxe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromFullInventory flat='false'/>
                        <ObservationFromRay/>  
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-'''+str(int(obs_size/2))+'''" y="44" z="-'''+str(int(obs_size/2))+'''"/>
                                <max x="'''+str(int(obs_size/2))+'''" y="44" z="'''+str(int(obs_size/2))+'''"/>
                            </Grid>
                        </ObservationFromGrid> 
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="'''+ str(50) + '''" yrange='46' zrange="'''+ str(50) + '''" />                                
                        </ObservationFromNearbyEntities>
                        <RewardForCollectingItem>
                            <Item reward="1" type="diamond"/>
                        </RewardForCollectingItem>
                        <RewardForTouchingBlockType>
                            <Block reward="-9999" type="cobblestone"/> 
                        </RewardForTouchingBlockType>
                        <AgentQuitFromReachingCommandQuota total="'''+str(max_episode_steps)+'''"/>
                        <MissionQuitCommands/>
                        <ChatCommands />
                        <RewardForSendingCommand reward="-1"/>
                        <RewardForSendingMatchingChatMessage>
                            <ChatMatch reward="100" regex="Collected all diamonds!" description="Anything that matches the object."/>
                        </RewardForSendingMatchingChatMessage>
                        <AgentQuitFromTouchingBlockType>
                            <Block type="cobblestone" />
                        </AgentQuitFromTouchingBlockType>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''

class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, actions=[], epsilon=0.1, alpha=0.1, gamma=1.0, debug=False, canvas=None, root=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = True

        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = actions
        self.q_table = {}
        self.canvas = canvas
        self.root = root
        
        self.rep = 0

        self.episode_step_arr = []
        self.episode_wins = []
        self.steps_taken = 0


    def loadModel(self, model_file):
        """load q table from model_file"""
        with open(model_file) as f:
            self.q_table = json.load(f)

    def training(self):
        """switch to training mode"""
        self.training = True


    def evaluate(self):
        """switch to evaluation mode (no training)"""
        self.training = False

    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.training and self.prev_s is not None and self.prev_a is not None:
            old_q = self.q_table[self.prev_s][self.prev_a]
            self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * (current_r + self.gamma * max(self.q_table[current_s]) - old_q)

        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l)-1)
            a = l[y]
            self.logger.info("Taking q action: %s" % self.actions[a])

        # send the selected action
        agent_host.sendCommand(self.actions[a])
        self.prev_s = current_s
        self.prev_a = a

        self.steps_taken = self.steps_taken + 1
        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        current_r = 0
        tol = 0.01
        
        self.prev_s = None
        self.prev_a = None
        
        # wait for a valid observation
        world_state = agent_host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = agent_host.peekWorldState()
        world_state = agent_host.getWorldState()
        for err in world_state.errors:
            print(err)

        if not world_state.is_mission_running:
            self.steps_taken = 0
            return 0 # mission already ended
            
        assert len(world_state.video_frames) > 0, 'No video frames!?'
        
        obs = json.loads( world_state.observations[-1].text )
        prev_x = obs[u'XPos']
        prev_z = obs[u'ZPos']
        print('Initial position:',prev_x,',',prev_z)
        
        if save_images:
            # save the frame, for debugging
            frame = world_state.video_frames[-1]
            image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
            iFrame = 0
            self.rep = self.rep + 1
            image.save( 'rep_' + str(self.rep).zfill(3) + '_saved_frame_' + str(iFrame).zfill(4) + '.png' )


        # take first action
        total_reward += self.act(world_state,agent_host,current_r)

        print("Mission Number: {}".format(mission_number))
        print("Steps: {}".format(self.steps_taken))

        grid = obs['entities']
        agent_pos = (grid[0]['x'], grid[0]['z']) # To make it easier to use agent positions

        touched = False
        dead = False

        # Check if agent is near a zombie
        for item in grid:
            if(item['name'] == 'Zombie'):
                touched, dead = self.is_near_entity(item['x'], item['z'], agent_pos[0], agent_pos[1])
                break

        if(touched == True):
            total_reward -= 1
        if(dead == True):
            total_reward -= 5
            print("Agent touched zombie!\n")
            self.agent_host.sendCommand("chat Agent died!")
            self.agent_host.sendCommand("quit")
            time.sleep(.2)
        
        diamonds_collected = 0
        inventory = [item for item in obs['inventory'] if item['type'] == 'diamond']
        if(len(inventory) != 0):
            diamonds_collected = inventory[0]['quantity']

        if(diamonds_collected == 31):
            self.episode_step_arr.append(self.steps_taken)
            self.episode_wins.append(mission_number)
            self.log_steps()
            agent_host.sendCommand("chat Collected all diamonds!")
            print("Collected all diamonds!\n")
            print("Steps taken: {}\n".format(self.steps_taken))
            agent_host.sendCommand("quit")
            time.sleep(.2)

        require_move = True
        check_expected_position = True
        
        # main loop:
        while world_state.is_mission_running:
        
            # wait for the position to have changed and a reward received
            print('Waiting for data...', end=' ')
            while True:
                world_state = agent_host.peekWorldState()
                if not world_state.is_mission_running:
                    print('mission ended.')
                    self.steps_taken = 0
                    break
                if len(world_state.rewards) > 0 and not all(e.text=='{}' for e in world_state.observations):
                    obs = json.loads( world_state.observations[-1].text )
                    #print("CHECKING")
                    curr_x = obs[u'XPos']
                    curr_z = obs[u'ZPos']
                    if require_move:
                        if math.hypot( curr_x - prev_x, curr_z - prev_z ) > tol:
                            print('received.')
                            break
                    else:
                        print('received.')
                        break
            # wait for a frame to arrive after that
            num_frames_seen = world_state.number_of_video_frames_since_last_state
            while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
                world_state = agent_host.peekWorldState()
                
            num_frames_before_get = len(world_state.video_frames)
            
            world_state = agent_host.getWorldState()
            for err in world_state.errors:
                print(err)
            current_r = sum(r.getValue() for r in world_state.rewards)

            if save_images:
                # save the frame, for debugging
                if world_state.is_mission_running:
                    assert len(world_state.video_frames) > 0, 'No video frames!?'
                    frame = world_state.video_frames[-1]
                    image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
                    iFrame = iFrame + 1
                    image.save( 'rep_' + str(self.rep).zfill(3) + '_saved_frame_' + str(iFrame).zfill(4) + '_after_' + self.actions[self.prev_a] + '.png' )
                
            if world_state.is_mission_running:
                assert len(world_state.video_frames) > 0, 'No video frames!?'
                num_frames_after_get = len(world_state.video_frames)
                assert num_frames_after_get >= num_frames_before_get, 'Fewer frames after getWorldState!?'
                frame = world_state.video_frames[-1]
                obs = json.loads( world_state.observations[-1].text )
                curr_x = obs[u'XPos']
                curr_z = obs[u'ZPos']
                print('New position from observation:',curr_x,',',curr_z,'after action:',self.actions[self.prev_a], end=' ') #NSWE
                if check_expected_position:
                    expected_x = prev_x + [0,0,-1,1][self.prev_a]
                    expected_z = prev_z + [-1,1,0,0][self.prev_a]
                    if math.hypot( curr_x - expected_x, curr_z - expected_z ) > tol:
                        print(' - ERROR DETECTED! Expected:',expected_x,',',expected_z)
                        input("Press Enter to continue...")
                    else:
                        print('as expected.')
                    curr_x_from_render = frame.xPos
                    curr_z_from_render = frame.zPos
                    print('New position from render:',curr_x_from_render,',',curr_z_from_render,'after action:',self.actions[self.prev_a], end=' ') #NSWE
                    if math.hypot( curr_x_from_render - expected_x, curr_z_from_render - expected_z ) > tol:
                        print(' - ERROR DETECTED! Expected:',expected_x,',',expected_z)
                        input("Press Enter to continue...")
                    else:
                        print('as expected.')
                else:
                    print()
                prev_x = curr_x
                prev_z = curr_z
                # act
                total_reward += self.act(world_state, agent_host, current_r)
                
        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.training and self.prev_s is not None and self.prev_a is not None:
            old_q = self.q_table[self.prev_s][self.prev_a]
            self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * ( current_r - old_q )
            
        self.drawQ()
    
        return total_reward
        
    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 20
        world_x = 21
        world_y = 21
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x,y)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale, 
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
                                     (curr_y + 0.5 - curr_radius ) * scale, 
                                     (curr_x + 0.5 + curr_radius ) * scale, 
                                     (curr_y + 0.5 + curr_radius ) * scale, 
                                     outline="#fff", fill="#fff" )
        self.root.update()


    def log_steps(self):

        #box = np.ones(self.log_frequency) / self.log_frequency
        #episode_smooth = np.convolve(self.episode_step_arr[1:], box, mode='same')
        plt.clf()
        plt.plot(self.episode_wins, self.episode_step_arr)
        plt.title('Pacman: Episodes v. Steps')
        plt.ylabel('Steps Taken')
        plt.xlabel('Episode No.')
        plt.savefig('pacman_steps.png')

        with open('pacman_steps.txt', 'w') as f:
            for step, value in zip(self.episode_wins, self.episode_step_arr):
                f.write("{}\t{}\n".format(step, value)) 

agent_host = MalmoPython.AgentHost()

# Find the default mission file by looking next to the schemas folder:

# add some args
agent_host.addOptionalFloatArgument('alpha','Learning rate of the Q-learning agent.', 0.1)
agent_host.addOptionalFloatArgument('epsilon','Exploration rate of the Q-learning agent.', 0.01)
agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 1.0)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')

malmoutils.parse_command_line(agent_host)

# -- set up the python-side drawing -- #
scale = 20
world_x = 21
world_y = 21
root = tk.Tk()
root.wm_title("Q-table")
canvas = tk.Canvas(root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
canvas.grid()
root.update()


if agent_host.receivedArgument("test"):
    num_maps = 1
else:
    num_maps = 30000

for imap in range(num_maps):

    # -- set up the agent -- #
    actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

    agent = TabQAgent(
        actions=actionSet,
        epsilon=agent_host.getFloatArgument('epsilon'),
        alpha=agent_host.getFloatArgument('alpha'),
        gamma=agent_host.getFloatArgument('gamma'),
        debug = agent_host.receivedArgument("debug"),
        canvas = canvas,
        root = root)

    # -- set up the mission -- #
    mission_xml = get_mission_xml()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.requestVideo( 1200, 900 )
    my_mission.setViewpoint( 1 )


    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    max_retries = 3
    agentID = 0
    expID = 'tabular_q_learning'

    num_repeats = 30000
    mission_number = 0

    cumulative_rewards = []
    for i in range(num_repeats):
        
        print("\nMap %d - Mission %d of %d:" % ( imap, i+1, num_repeats ))
        mission_number = mission_number + 1

        my_mission_record = malmoutils.get_default_recording_object(agent_host, "./save_%s-map%d-rep%d" % (expID, imap, i))

        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i) )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2.5)

        print("Waiting for the mission to start", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
        print()
        time.sleep(2)
        # -- run the agent in the world -- #
        cumulative_reward = agent.run(agent_host)
        print('Cumulative reward: %d' % cumulative_reward)
        cumulative_rewards += [ cumulative_reward ]

        # -- clean up -- #
        time.sleep(1) # (let the Mod reset)

    print("Done.")

    print()
    print("Cumulative rewards for all %d runs:" % num_repeats)
    print(cumulative_rewards)
