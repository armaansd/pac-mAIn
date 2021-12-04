---
layout: default
title:  Home
---

# PROJECT SUMMARY

<p>The goal of our project is to train our agent to play a modified Mincraft recreation of the game Pac-Man. The goal of our agent is to get as many points as possible before time runs out (500 steps). The agent will be placed in a 21 x 21 enclosed maze and will have to traverse the map in order to pick up diamonds located around the maze. There will be a total of 35 diamonds for the agent to collect. The agent's score will be based on how many diamonds they collect in a given episode. We will develop our AI using Malmo. </p>

<p>In this maze will also include a zombie, replacing the ghost in the original Pac-Man game. The agent will have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie and touches it, it will receive a penalty score.</p>

<p>The goal of this project is to implement the environment ourselves and implement and compare more than one algorithm. </p>


<img src="https://user-images.githubusercontent.com/75513952/144721188-2a29f8d2-261c-4e87-98e4-140f43b1356b.png" width="1000" height="700">
<img src="https://user-images.githubusercontent.com/75513952/144721196-a181fdb6-e224-46f8-89c8-f8b841001e1e.png" width="700" height="500">
<img src="https://user-images.githubusercontent.com/75513952/144721214-fc78789b-1386-41cd-9333-d64194b9af1b.png" width="700" height="500">
<img src="https://user-images.githubusercontent.com/75513952/144721228-0c75c8dd-959e-4998-82a2-fb055386e1da.png" width="700" height="500">


## Source code: 
- <https://github.com/armaansd/pac-main>

## Reports:
- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)

## Algorithms We Will Explore
- PPO
- Tabular Q-Learning

## Resources Used 

- <https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html>
- <https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo>   
- <https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py>    
- <https://minecraft-archive.fandom.com/wiki/Blocks>


[quickref]: https://github.com/mundimark/quickrefs/blob/master/HTML.md
