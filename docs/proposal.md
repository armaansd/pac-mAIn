---
layout: default
title: Proposal
---

![image](https://user-images.githubusercontent.com/75513952/138029586-91aba8ae-47cb-4680-a669-72810dc42a33.png)




## 1. Project Summary
   
<p>Pac-Man is action maze game where the player navigates through a maze to collect pellets in order to score points. The player does this while evading enemies located in the maze. The goal of our project is to train an agent to play a Minecraft recreation of the game. The recreation will be similar to the original game, except it will be watered down. The goal of our agent is to be able to collect most or all of the pellets while having the minimum amount of deaths per run (3 or less). The agent will also need to learn to avoid enemies. The input to our agent could be the map of the maze and the items and enemies within the maze. The output of our agent will be the movement list the agent will perform.</p>
   
   
   
   
## 2. AI/ML Algorithms

<p>We are planning to implement Deep-Q Learning and reinforcement learning to train our agent.</p>




## 3. Evaluation Plan

<p>Because the goal of Pac-Man is to gather points, we will evaluate it based on this metric. Our agent will receive plus points for collecting pellets and will receive negative points for coming into contact with an enemy. The baseline is for the agent to be able to score a predetermined amount of points in the maze (we will decide this later on). We can further evaluate on this based on how long it takes the agent to complete this task as well how many deaths does it take before it can complete its task.</p> 

<p>There are several sanity checks we can do to qualitatively access our agent. We can observe if our agent is avoiding enemies. It is considered a failure if our agent constantly runs into enemies and loses all of its lives. We expect the agent to lose some lives, but not lose the game. Likewise, the agent should be actively looking for pellets to collect. If the agent is just standing around for extended periods of time, then we will consider it a failure.</p>




## 4. Appointment with the Instructor
- 3:15pm - 3:30pm, Tuesday, October 26, 2021

