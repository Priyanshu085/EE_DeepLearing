import gym
import numpy as np

q_table = np.zeros((state_size, action_size))
alpha = 0.8
gamma = 0.95

for episode in range(num_episodes):
	state = gym.Env.reset()
	done = False
	while not done:
		# Choose an action
		action = np.argmax(
			q_table[state, :] + np.random.randn(1, action_size) * (1. / (episode + 1)))

		# Take the action and observe the new state and reward
		next_state, reward, done, _ = env.step(action)

		# Update the Q-table
		q_table[state, action] = (1 - alpha) * q_table[state, action] + \
			alpha * (reward + gamma * np.max(q_table[next_state, :]))

		state = next_state

# Test the trained Q-Learning algorithm
state = env.reset()
done = False
while not done:
	# Choose an action
	action = np.argmax(q_table[state, :])

	# Take the action
	state, reward, done, _ = env.step(action)
	env.render()
