# Reading Course - POMDP
During the fall of 2020 I did a 7.5 Credit reading course about POMDP:s which included a theoretical study of the field and code implementations of both filtering algorithms and Deep Reinforcment Learning algorithms. 

This repo includes: 
* Implementtions of a DQN [[1]](#1), DQRN [[2]](#2) and a network inspiered by ADRQN [[3]](#3) using pytorch. The different models are then trained and used to solve the CartPole-problem from the OpenAI Gym, both as a MDP and a POMDP.

* A presentation about how POMDP:s have been solved theoreticly before Deep RL. 

## References
<a id="1">[1]</a> 
[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), Mnih et al, 2013.

<a id="2">[2]</a> 
[Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527), Hausknecht and Stone, 2015.

<a id="3">[3]</a> 
[On Improving Deep Reinforcement Learning for POMDPs](https://arxiv.org/abs/1704.07978), Zhu et al, 2018.

