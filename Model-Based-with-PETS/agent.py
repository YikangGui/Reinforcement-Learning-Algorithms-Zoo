import numpy as np
import time

np.random.seed(1)


class Agent:
    def __init__(self, env, noisy=False):
        self.env = env
        self.noisy = noisy
        print('env:', self.env, 'noise:', self.noisy)

    def sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        # for t in range(horizon):
        t = 0
        ts1 = time.time()
        while t < horizon:
            # print('time step: {}/{}'.format(t, horizon))
            # print(f"time step{t}/{horizon}")
            if policy.use_mpc:
                actions.append(policy.act(states[t], t, self.noisy))
                state, reward, done, info = self.env.step(actions[t])
                states.append(state)
                reward_sum += reward
                rewards.append(reward)
                t += 1
            else:
                actions_tmp = policy.act(states[t], t)
                for i in range(len(actions_tmp)):
                    actions.append(actions_tmp[i])
                    state, reward, done, info = self.env.step(actions_tmp[i])
                    states.append(state)
                    reward_sum += reward
                    rewards.append(reward)
                    t += 1
                    if t >= horizon:
                        break
                    if done:
                        break

            if done:
                print(info['done'])
                break

        print("Rollout length: ", len(actions), 'Time: %.2f' % (time.time() - ts1))
        print()

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }


class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.use_mpc = False

    def reset(self):
        pass

    def act(self, arg1, arg2):
        return [np.random.uniform(size=self.action_dim) * 2 - 1]
