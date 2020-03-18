This repository contains 2 Python scripts for solving the LunarLander-v2 openai gym
environment using deep reinforcement learning algorithms: REINFORCE algorithm (without baseline) is found in reinforce.py; 
N-step Advantage Actor-Critic algorithm found in a2c.py

Note: current implementation of a2c.py contains critic network model parameters
specific to n=1 a2c algorithm (namely, 30x30x30 MLP instead of 20x20x20 MLP
used for n=20, 50, and 100).

```
usage: reinforce.py [-h] [--model-config-path MODEL_CONFIG_PATH]
                    [--num-episodes NUM_EPISODES] [--lr LR]
                    [--render | --no-render]

optional arguments:
  -h, --help            show this help message and exit
  --model-config-path MODEL_CONFIG_PATH
                        Path to the model config file.
  --num-episodes NUM_EPISODES
                        Number of episodes to train on.
  --lr LR               The learning rate.
  --render              Whether to render the environment.
  --no-render           Whether to render the environment.
```
=========================================================
```
usage: a2c.py [-h] [--model-config-path MODEL_CONFIG_PATH]
              [--num-episodes NUM_EPISODES] [--lr LR] [--critic-lr CRITIC_LR]
              [--n N] [--render | --no-render]

optional arguments:
  -h, --help            show this help message and exit
  --model-config-path MODEL_CONFIG_PATH
                        Path to the actor model config file.
  --num-episodes NUM_EPISODES
                        Number of episodes to train on.
  --lr LR               The actor's learning rate.
  --critic-lr CRITIC_LR
                        The critic's learning rate.
  --n N                 The value of N in N-step A2C.
  --render              Whether to render the environment.
  --no-render           Whether to render the environment.

```
