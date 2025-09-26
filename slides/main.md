---
title:

- Reinforcement Learning for Deliberation in ROS 2
author:
- Christian Henkel
- Sebastian Castro
theme:
- Bergen
date:
- ROSCon 2025 / October 27, 2025
logo:
- ros-wg-delib.png
aspectratio: 169
---

# Introduction
<!-- Build with `pandoc -t beamer main.md -o main.pdf` -->

What is RL?

![Agent Environment Interaction](https://media.geeksforgeeks.org/wp-content/uploads/20240516130928/agent-environment-interface-(3).png)

# Test your setup

```bash
pixi run pyrobosim_demo
```

# Basics: MDP

A Markov Decision Process (MDP) is defined as $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$.

- $\mathcal{S}$ States.
- $\mathcal{A}$ Possible Actions.
- ...

# Basics: Belman Equation

$V(s) = \max_a {   }$.
