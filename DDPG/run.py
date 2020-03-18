import gym
import envs
from algo.ddpg import DDPG, TD3, TD3_Torch


def main():
    env = gym.make('Pushing2D-v0')
    # env = gym.make('Pendulum-v0')
    env.seed(1024)
    algo = DDPG(env, 'ddpg_log.txt')
    # algo = TD3(env, 'ddpg_log.txt')
    # algo = TD3_Torch(env, 'ddpg_log.txt')

    algo.train(50000, hindsight=True)
    # algo.train(50000, hindsight=False)


if __name__ == '__main__':
    main()
