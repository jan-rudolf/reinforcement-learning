# Monte Carlo ES (Exploting Starts)
# from page 99 of Reinforcement Learning by R. Sutton
#
# Author: Jan Rudolf

import gym
import numpy as np


def create_key(state, action):
	return tuple(state.tolist().append(action))


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
		self._returns = dict()
		self.value = 0

	def _key(self, state, action):
		key = tuple(state.tolist().append(action))
		if key not in self._returns:
			self._returns[key] = list()
		return key

	def add_return(self, state, action, g):
		key = self._key(state, action)
		self._returns[key].append(g)

	def average_return(self, state, action):
		key = self._key(state, action)
		if len(self._returns[key]):
			return np.average(self._returns[key])
		return 0

if __name__ == '__main__':
	env = gym.make('CartPole-v1')

	epsilon = 0.2

	policy = Policy(env.action_space)
	q = Q()

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

		#reward sum
		g = 0
		gamma = 0.9

		history_went_through = []
		# update action-value function and policy
		for history_item in reversed(history):
			state, action, reward = history_item

			key = create_key(state, action)
			history_went_through.append(key)

			g = gamma*g + reward

			if key not in  history_went_through:
				q.add_return(state, action, g)
