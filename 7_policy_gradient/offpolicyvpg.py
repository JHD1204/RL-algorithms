import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from vpg import VPGAgent

env = gym.make('MountainCar-v0')

class OffPolicyVPGAgent(VPGAgent):
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.trajectory = []

        def dot(y_true, y_pred):
          return -tf.reduce_sum(y_true * y_pred, axis=-1)

        self.policy_net = self.build_network(output_size=self.action_n, output_activation=tf.nn.softmax,
                                             loss=dot, **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1, **baseline_kwargs)

    def learn(self, observation, action, behavior, reward, done):
        self.trajectory.append((observation, action, behavior, reward))
        if done:
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'behavior', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_reward']

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):  # 判断对象是否包含对应的属性
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= (df['baseline'] * df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)
            y = np.eye(self.action_n)[df['action']] * (df['psi'] / df['behavior']).values[:, np.newaxis]
            self.policy_net.fit(x, y, verbose=0)
            self.trajectory = []

policy_kwargs = {'hidden_sizes': [10,], 'activation': tf.nn.relu, 'learning_rate': 0.01}
agent = OffPolicyVPGAgent(env, policy_kwargs=policy_kwargs)

class RandomAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n

    def decide(self, observation):
        action = np.random.choice(self.action_n)
        behavior = 1. / self.action_n
        return action, behavior

behavior_agent = RandomAgent(env)

def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward

episodes = 1500
episode_rewards = []
for episode in range(episodes):
    observation = env.reset()
    episode_reward = 0.
    while True:
        action, behavior = behavior_agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.learn(observation, action, behavior, reward, done)
        if done:
            break
        observation = next_observation
    # 跟踪监控
    episode_reward = play_montecarlo(env, agent)
    episode_rewards.append(episode_reward)

plt.plot(episode_rewards)
plt.show()

episode_rewards = [play_montecarlo(env, agent) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))








