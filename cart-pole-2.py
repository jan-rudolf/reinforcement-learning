# Monte Carlo ES (Exploting Starts)
# from page 99 of Reinforcement Learning by R. Sutton
#
# Author: Jan Rudolf
import time

import gym
import numpy as np


def create_key(state, action):
	state = state.tolist()
	state.append(action)
	return tuple(state)


class Policy:
	def __init__(self, action_space):
		self._state_action_list = dict()
		self.action_space = action_space

	def set_state_action(self, state, action):
		key = tuple(state.tolist())
		self._state_action_list[key] = action

	def act(self, state):
		state_key = tuple(state.tolist())

		if state_key in self._state_action_list:
			return self._state_action_list[state_key]
		else:
			action = self.action_space.sample()
			self._state_action_list[state_key] = action
			return action


class Q:
	def __init__(self):
		self._returns = dict()
		self._values = dict()

	def _key(self, state, action):
		state = state.tolist()
		state.append(action)
		key = tuple(state)
		if key not in self._returns:
			self._returns[key] = list()
			self._values[key] = 0
		return key

	def add_return(self, state, action, g):
		key = self._key(state, action)
		self._returns[key].append(g)

	def average_return(self, state, action):
		key = self._key(state, action)
		if len(self._returns[key]):
			self._values[key] = np.average(self._returns[key])
		self._values[key] = 0

	def value(self, state, action):
		key = self._key(state, action)
		return self._values[key]

if __name__ == '__main__':
	env = gym.make('CartPole-v1')

	epsilon = 0.2

	policy = Policy(env.action_space)
	q = Q()

	for i_episode in range(50):
		history = []

		observation = env.reset()

		# generate an episode
		for i in range(1000):
			env.render()
			time.sleep(0.1)

			action = policy.act(observation) # env.action_space.sample()
			observation, reward, done, info = env.step(action)

			print("iteration: ", i)
			print("\tobservation: ", observation)
			print("\taction: ", action)
			print("\treward: ", reward)

			history.append((observation, action, reward))

			if done:
				print("Episode {} finished after {} timesteps".format(i_episode, i+1))
				break

		#reward sum
		g = 0
		gamma = 0.9

		history_went_through = []
		# update action-value function and policy
		for history_item in reversed(history):
			state, action, reward = history_item

			key = create_key(state, action)


			g = gamma*g + reward

			if key not in  history_went_through:
				q.add_return(state, action, g)
				q.average_return(state, action)

				if q.value(state, 0) < q.value(state, 1):
					policy.set_state_action(state, q.value(state, 1))
				else:
					policy.set_state_action(state, q.value(state, 0))
				history_went_through.append(key)
