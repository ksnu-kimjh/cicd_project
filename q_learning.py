import gymnasium as gym
import numpy as np
import argparse
import pickle

def train_agent(episodes, learning_rate, gamma, epsilon):
	# Env creation
	env = gym.make('FrozenLake-v1', is_slippery=False)
	q_table = np.zeros((env.observation_space.n, env.action_space.n))

	for _ in range(episodes):
		state, _ = env.reset()
		done = False
		while not done:
			# epsilon-greedy strategy
			if np.random.uniform(0, 1) < epsilon:
				action = env.action_space.sample()
			else:
				action = np.argmax(q_table[state, :])
			
			next_state, reward, done, _, _ = env.step(action)
			# Q-learning update
			q_table[state, action] = q_table[state, action] + learning_rate * \
				(reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
			state = next_state
	return q_table

if __name__ == '__main__':
	print(">>> [System] Model optimizatino under Github Actions!")
	parser = argparse.ArgumentParser()
	parser.add_argument("--episodes", type=int, default=1000)
	parser.add_argument("--lr", type=float, default=0.8)
	parser.add_argument("--gamma", type=float, default=0.95)
	args = parser.parse_args()

	model = train_agent(args.episodes, args.lr, args.gamma, 0.1)

	with open("q_table.pkl", "wb") as f:
		pickle.dump(model, f)
	print(">>> [System] Training completed and model saved (q_table.pkl)")

