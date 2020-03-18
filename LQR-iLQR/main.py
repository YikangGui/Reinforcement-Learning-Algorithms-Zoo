import gym
import numpy as np
from deeprl_hw6.arm_env import TwoLinkArmEnv
from controllers import calc_lqr_input
from ilqr2 import calc_ilqr_input


def lqr():
	env = gym.make('TwoLinkArm-v0')
	sim_env = gym.make('TwoLinkArm-v0')
	x = env.reset()
	u_old = np.ones((2,))
	total_reward, step = 0, 0	
	q_trajectory, dq_trajectory, u_trajectory = [], [], []
	q_trajectory.append(x[:2])
	dq_trajectory.append(x[2:])
	while True:
		u_new = calc_lqr_input(env, sim_env, u_old)
		x, reward, is_done, _ = env.step(u_new)
		print("###")
		print("u: "+ str(u_new))
		print("x: "+ str(x))
		# u_old = u_new
		total_reward += reward
		step += 1
		print('total reward: {}, step: {}'.format(total_reward, step))
		q_trajectory.append(x[:2])
		dq_trajectory.append(x[2:])
		u_trajectory.append(u_new)
		if is_done:
			break
		if step==1000:
			break

def ilqr():
	env = gym.make('TwoLinkArm-v0')
	sim_env = gym.make('TwoLinkArm-v0')
	x = env.reset()
	# u_old = np.ones((2,))
	total_reward, step = 0, 0	
	q_trajectory, dq_trajectory, u_trajectory = [], [], []
	q_trajectory.append(x[:2])
	dq_trajectory.append(x[2:])
	while True:
		U = calc_ilqr_input(env, sim_env, tN=100)
		u_new = U[0]
		x, reward, is_done, _ = env.step(u_new)
		print("###")
		print("u: "+ str(u_new))
		print("x: "+ str(x))
		# u_old = u_new
		total_reward += reward
		step += 1
		print('total reward: {}, step: {}'.format(total_reward, step))
		q_trajectory.append(x[:2])
		dq_trajectory.append(x[2:])
		u_trajectory.append(u_new)
		if is_done:
			break
		# if step==1000:
		# 	break

if __name__ == "__main__":
    # lqr()
    ilqr()
