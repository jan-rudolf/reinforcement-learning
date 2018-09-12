# Monte Carlo ES (Exploting Starts)
# from page 99 of Reinforcement Learning by R. Sutton
#
# Author: Jan Rudolf

import gym

class Policy:
	def __init__(self, action_space):
		self._state_action_list = dict()
		self.action_space = action_space

	def act(self, state):
		state_key = tuple(state.tolist())

		if state_key in self._state_action_list:
			pass
		else:
			action = self.action_space.sample()
			self._state_action_list[state_key] = action
			return action


class Q:
	def __init__(self):
		pass


if __name__ == '__main__':
	env = gym.make('CartPole-v1')

	epsilon = 0.2

	policy = Policy(env.action_space)

	for i_episode in range(20):
		history = []

		observation = env.reset()

		# generate an episode
		for i in range(1000):
			env.render()

			action = policy.act(observation) # env.action_space.sample()
			observation, reward, done, info = env.step(action)

			print("iteration: ", i)
			print("\tobservation: ", observation)
			print("\taction: ", action)
			print("\treward: ", reward)

			history.append((observation, action, reward))

			if done:
				print("Episode finished after {} timesteps".format(i+1))
				break

		# update action-value function and policy
