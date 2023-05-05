import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras

env = gym.make('CartPole-v0')
env.seed(0)

class VPGAgent:
    def __init__(self, env, policy_kwargs, baseline_kwargs=None,
                 gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.trajectory = []

        self.policy_net = self.build_network(output_size=self.action_n,
                                             output_activation=tf.nn.softmax,
                                             loss=tf.losses.categorical_crossentropy,
                                             **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1,
                                                   **baseline_kwargs)

    def build_network(self, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for hidden_size in hidden_sizes:
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        probs = self.policy_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory,
                              columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            '''
            print('discount={}'.format(df['discount']))
            print('reward={}'.format(df['reward']))
            print('discounted_reward={}'.format(df['discounted_reward']))
            print('discounted_return={}'.format(df['discounted_return']))
            '''

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= (df['baseline'] * df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            y = np.eye(self.action_n)[df['action']] * \
                df['psi'].values[:, np.newaxis]
            self.policy_net.fit(x, y, verbose=0)

            self.trajectory = []  # 为下一回合初始化经验列表

def play_montecarlo(env, agent, render=False, train=False):
    observation = env.reset()
    episode_reward = 0.
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

policy_kwargs = {'hidden_sizes' : [10,], 'activation' : tf.nn.relu,
        'learning_rate' : 0.008}
agent = VPGAgent(env, policy_kwargs=policy_kwargs)

# 训练
episodes = 500
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_montecarlo(env, agent, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()

# 测试
episode_rewards = [play_montecarlo(env, agent, train=False)
        for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))