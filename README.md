# Reinforcement Learning for Deliberation in ROS 2

This repository contains materials for the [ROSCon 2025](https://roscon.ros.org/2025/) workshop on ROS 2 Deliberation Technologies.

## Setup

This repo uses Pixi and RoboStack along with ROS 2 Kilted.

First, clone the repo including submodules.

```bash
git clone --recurse-submodules https://github.com/ros-wg-delib/roscon25-workshop.git
```

Then, install Pixi.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Build the environment.

```bash
pixi run build
```

Launch an example demo.

```bash
pixi run pyrobosim_demo
```

You can also drop into a shell in the Pixi environment.

```bash
pixi shell
```

## Training a model

```bash
pixi run start_world --headless
```

```bash
pixi run train --env PickBanana --model-type PPO --log
```

```bash
pixi run tensorboard
```

```bash
pixi run eval --model <path_to_model.pt>
```
