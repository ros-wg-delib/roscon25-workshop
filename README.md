# Reinforcement Learning for Deliberation in ROS 2

This repository contains materials for the [ROSCon 2025](https://roscon.ros.org/2025/) workshop on ROS 2 Deliberation Technologies.

## Setup

This repo uses Pixi and RoboStack along with ROS 2 Kilted.

First, clone the repo including submodules.

<!--- new-env: ubuntu:latest --->
<!--
```bash
apt update
apt install -y git curl build-essential
```
-->

```bash
git clone --recurse-submodules https://github.com/ros-wg-delib/roscon25-workshop.git
```

Then, install Pixi.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

<!--
This is necessary to make pixi work ...
```bash
echo 'export PATH=\"/root/.pixi/bin:$PATH\"' >> /root/.bashrc
```
-->

Build the environment.

<!--- workdir: /roscon25-workshop --->
```bash
pixi run build
```

Launch an example demo.

<!--- skip-next --->
```bash
pixi run pyrobosim_demo
```

You can also drop into a shell in the Pixi environment.

<!--- skip-next --->
```bash
pixi shell
```

## Training a model

### Start Environment

<!--- skip-next --->
```bash
pixi run start_world --headless
```

### Choose modely type

For example PPO

<!--- skip-next --->
```bash
pixi run train --env PickBanana --model-type PPO --log
```

Or DQN.
Note that this needs the `--discrete-actions` flag.

<!--- skip-next --->
```bash
pixi run train --env PickBanana --model-type DQN --discrete-actions --log
```

### You may find tensorboard useful

<!--- skip-next --->
```bash
pixi run tensorboard
```

### See your freshly trained policy in action

<!--- skip-next --->
```bash
pixi run eval --model <path_to_model.pt>
```
