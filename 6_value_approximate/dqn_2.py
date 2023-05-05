import gym
import numpy as np
import matplotlib.pyplot as plt
from dqnreplayer import DQNReplayer
import tensorflow as tf
from tensorflow import keras

env = gym.make('MountainCar-v0')
env = env.unwrapped


class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
                 replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)  # 经验回放

        self.evaluate_net = self.build_network(input_size=observation_dim,
                                               output_size=self.action_n, **net_kwargs)  # 评估网络
        self.target_net = self.build_network(input_size=observation_dim,
                                             output_size=self.action_n, **net_kwargs)  # 目标网络

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      learning_rate=0.01):  # 构建网络
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))  # 输出层
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)  # 存储经验

        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)  # 经验回放

        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:  # 更新目标网络
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation):  # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)

def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation,
                    done)
        if done:
            break
        observation = next_observation
    return episode_reward

net_kwargs = {'hidden_sizes' : [16,], 'learning_rate' : 0.01}
agent = DQNAgent(env, net_kwargs=net_kwargs)

# 训练
episodes = 500
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
    print('episode {} game over'.format(episode))
plt.plot(episode_rewards)

# 测试
agent.epsilon = 0. # 取消探索
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))