---
layout: default
title:  Home
---

# PROJECT SUMMARY
The goal of our project is to train our agent to play a modified version of Pac-Man on Minecraft. We will develop our AI using the Malmo platform. The agent will be placed in a maze to explore. The pellets from the original game will be substituted with diamonds and the ghosts will be substituted by a Zombie. The agent's score will be based on how many diamonds they collect in a given episode. The agent will "win" if it is able to collect all diamonds.

Of course, this project would be trivial if the agent only needed to collect diamonds. The agent will also have to learn to avoid the zombie as it picks up diamonds. If it gets near a zombie it will receive a penalty score. If it gets touched by the zombie, the agent will "die" and received a larger penalty score. Thus, the agent will have to learn to maneuver itself with this in mind. Because the position of the zombie changes over time and the diamonds disappear when they are collected, ML and/or RL algorithms can help the agent make reasonable actions in an environment with dynamic entities. It will be interesting to see if our agent will develop strategies for collecting diamonds similar to how a human might do.

We will create the environment ourselves and train our agent using two different approaches. We will then evaluate our agent based on several metrics.


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


[quickref]: https://github.com/mundimark/quickrefs/blob/master/HTML.md
