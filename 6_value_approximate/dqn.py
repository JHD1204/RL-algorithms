import gym
import numpy as np
import matplotlib.pyplot as plt
from dqnreplayer import DQNReplayer
import tensorflow as tf
from tensorflow import keras

env = gym.make('MountainCar-v0')
env = env.unwrapped

class DQNAgent:
    def __init__(self, env, net_kwargs, gamma=0.99, epsilon=0.001,
                 replayer_capacity=10000, batch_size=32):
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
                      learning_rate=0.01):  # 构建网络, 学习率为0.01
        model = keras.Sequential()
        '''
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, **kwargs))
        '''
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}     # 创建字典
            if idx ==0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation))  # 输出层
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self):
        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)               # 经验回放  next_observations ndarray(32, 2)
        next_qs = self.target_net.predict(next_observations)    # ndarray(32, 3)
        next_max_qs = next_qs.max(axis=-1)                      # ndarray(32, )
        us = rewards + self.gamma * (1. - dones) * next_max_qs      # ndarray(32, )
        targets = self.evaluate_net.predict(observations)           # ndarray(32, 3)
        targets[np.arange(us.shape[0]), actions] = us               # 仅改变actions对应的target值
        self.evaluate_net.fit(observations, targets, verbose=0)

    def decide(self, observation):  # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)

net_kwargs = {'hidden_sizes' : [16,], 'learning_rate' : 0.01}
agent = DQNAgent(env, net_kwargs=net_kwargs)

# 训练
episodes = 500
episode_rewards = []
step_min = 100
step = 0
render = False
train = True
for episode in range(episodes):
    observation = env.reset()
    episode_reward = 0
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        agent.replayer.store(observation, action, reward, next_observation, done)  # 存储经验
        episode_reward += reward
        if train and (step > step_min):
            agent.learn()
            if done:  # 更新目标网络
                agent.target_net.set_weights(agent.evaluate_net.get_weights())
        observation = next_observation
        if done:
            break
        step += 1
    episode_rewards.append(episode_reward)
    print('episode {} game over'.format(episode))
plt.plot(episode_rewards)
plt.show()

# 测试
agent.epsilon = 0. # 取消探索
episodes_test = 100
episode_rewards_test = []
for episode in range(episodes_test):
    observation = env.reset()
    episode_reward = 0
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        observation = next_observation
        if done:
            break
    episode_rewards_test.append(episode_reward)
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards_test),
        len(episode_rewards_test), np.mean(episode_rewards_test)))
