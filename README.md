# Quasimetric Reinforcement Learning

*Master Thesis by Patrick Siebke, TU Darmstadt – November 15, 2024*

## Overview
This repository contains the code, experiments, and supporting materials from my master thesis on **Quasimetric Reinforcement Learning (QRL)**. The thesis addresses two central research questions in Goal-Conditioned Reinforcement Learning (GCRL):

1. **Impact of a Quasimetric Critic:**  
   How does modeling the optimal value function as a quasimetric affect learning? I compare a standard neural network critic with a quasimetric critic based on **Interval Quasimetric Embedding (IQE)**, as well as a metric critic derived from IQE to enforce symmetry.

2. **Extended Goal Relabeling:**  
   Can integrating random off-trajectory goal relabeling with on-trajectory relabeling in Hindsight Experience Replay (HER) introduce a pessimism bias that improves both robustness and sample efficiency?

Experiments on robotics tasks from the Gymnasium Robotics suite—specifically Fetch and HandManipulate environments—demonstrate that these approaches significantly enhance learning performance.

## Introduction
### Challenges in GCRL
In goal-conditioned reinforcement learning, agents learn to achieve multiple goals from sparse reward signals. Two key challenges are:

- **Sparse Rewards:** Feedback is only provided when a goal is reached, which hinders efficient learning.
- **Temporal Distance as a Quasimetric:** The optimal value function, which represents the minimum cost or “distance” to reach a goal, is inherently asymmetric. This asymmetry—where the cost from state A to B may differ from the cost from B to A—is best modeled using a quasimetric.

### Contributions
In this work, I make the following contributions:
- **Quasimetric Critic Architectures:**  
  - A conventional neural network critic (symmetric).
  - A quasimetric critic using **Interval Quasimetric Embedding (IQE)** to capture directional costs.
  - A metric critic (Interval Metric Embedding, IME) that enforces symmetry by averaging directional costs.
  
- **Extended HER Relabeling Strategies:**  
  I propose mixed relabeling strategies that decouple the relabeling for the actor and the critic, combining random off-trajectory relabeling with traditional on-trajectory relabeling. This mixed strategy introduces a pessimism bias that aids the critic in generalizing to unseen goals.

## Methodology
### Quasimetric Reinforcement Learning
The core idea is to approximate the optimal value function \( V^*(s, g) \) as a quasimetric \( d_\theta(s, g) \). This formulation naturally enforces:
- **Triangle Inequality:** The cost of any detour is at least as high as that of the direct transition.
- **Asymmetry:** It accurately captures that moving toward a goal can incur a different cost than moving away.

To implement this, the thesis explores:
- **Interval Quasimetric Embedding (IQE):** A latent embedding that satisfies the properties of a quasimetric.
- **Metric Residual Networks:** An approach where a symmetric metric is derived from the quasimetric by averaging the directional costs.

### Extended Hindsight Experience Replay (HER)
HER is used to mitigate sparse rewards by relabeling goals based on future states. In my work, I extend HER by:
- Decoupling the relabeling process for the actor and the critic.
- Introducing **random off-trajectory relabeling** in conjunction with on-trajectory relabeling, which imposes a pessimism bias beneficial for learning robust value functions.

### Tasks & Environments
The experimental evaluation is conducted on:
- **Fetch Environments:** Including FetchPush, FetchSlide, and FetchPickAndPlace—tasks that involve pushing, sliding, and pick-and-place operations with a 7-DoF robotic arm.
- **HandManipulate Tasks:** Involving complex manipulation with a simulated Shadow Dexterous Hand.

## Results
The experimental results indicate that:
- **Sample Efficiency is Improved:** Quasimetric critics based on IQE significantly outperform standard neural network critics.
- **Learning Robustness is Enhanced:** Mixed relabeling strategies that combine random and future goal relabeling yield more stable performance.
- **Plots:**  
  *TODO add plots and videos*

## Installation & Usage

### Clone the Repository
```sh
git clone https://github.com/pynator/QuasimetricRL.git
```

### Installation Options

#### Option 1: Using Docker
A Dockerfile is provided for a reproducible environment. To build and run the container use:

```
# Build the Docker image
docker build -t qrl .

# Run the Docker container
docker run -it --rm --gpus all -p 6006:6006 -v {path_to_project}/QuasimetricRL:/workspace qrl /bin/bash
```

#### Option 2: Manual Installation
If you prefer not to use Docker, you can install the required libraries using pip:

```
pip install torch
pip install mujoco==3.1.6
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git
pip install torch-tb-profiler
pip install moviepy
pip install wandb
```

### Running the Experiments
To train a policy you can use the run.sh script.
```
./run.sh
```


### Visualizing the trained policy
To generate a video of the trained policy, use the video.sh script.
```
./video.sh
```

## References
[Interval Quasimetric Embedding (IQE) (Paper)](https://arxiv.org/abs/2211.15120)

[Interval Quasimetric Embedding (IQE) (Code - torchqmet)](https://github.com/quasimetric-learning/torch-quasimetric)

[Metric Residual Network (Code)](https://github.com/Cranial-XIX/metric-residual-network)

[Gymnasium Robotics](https://robotics.farama.org/index.html)