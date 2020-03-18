import matplotlib.pyplot as plt
import numpy as np

episode_path = '_episode.npy'
mean_path = '_mean.npy'
std_path = '_std.npy'

with open('nohup.out', 'r') as f:
    for n in [20, 50]:
        tmp_mean, tmp_std = [], []
        tmp_episode = np.load(str(n) + episode_path)
        line = f.readline()
        while line:
            line = line.split()
            if 'Test' in line and 'reward' in line:
                tmp_mean.append(float(line[4]) + 10)
                tmp_std.append(float(line[7]))
            if line == ['Test', 'Completed!']:
                break
            line = f.readline()

        plt.errorbar(tmp_episode, tmp_mean, tmp_std, capsize=2)
        plt.xlabel('Episode')
        plt.ylabel("Reward")
        plt.savefig(str(n) + '.png')
        plt.clf()
