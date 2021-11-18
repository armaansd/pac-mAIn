---
layout: default
title:  Home
---

# PROJECT SUMMARY

<p>The goal of our project is to train our agent to play a modified Mincraft recreation of the game Pac-Man. The goal of our agent is to get as many points as possible before time runs out (500 steps). The agent will be placed in a 28 x 31 enclosed maze and will have to traverse the map in order to pick up diamonds located around the maze. The agent's score will be based on how many diamonds they collect in a given episode. We will develop our AI using Malmo. </p>

<p>In this maze will also include a zombie, replacing the ghost in the original Pac-Man game. The agent will have to learn to avoid the zombie as it picks up diamonds. If it encounters a zombie and touches it, it will receive a penalty score.</p>

<img src="https://user-images.githubusercontent.com/75513952/142333894-bb3948dc-c27e-4b5d-9723-6c287426d49d.png" width="1000" height="700">

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
