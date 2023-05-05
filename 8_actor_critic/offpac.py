import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import keras
from keras import layers

env = gym.make('Acrobot-v1')
env.seed(0)

# 异策算法
class OffPACAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.
        self.critic_learning_rate = critic_kwargs['learning_rate']
        self.actor_net = self.build_network(output_size=self.action_n,
                                            output_activation=tf.nn.softmax, **actor_kwargs)
        self.critic_net = self.build_network(output_size=self.action_n,
                                             **critic_kwargs)

    def build_network(self, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))
        optimizer = tf.optimizers.SGD(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, behavior, reward,
              next_observation, done):
        observations = np.float32(observation[np.newaxis])
        pi = self.actor_net(observations)[0, action]  # 用于训练critic
        q = self.critic_net(observations)[0, action]  # 用于训练actor

        # 训练actor
        x_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)
            loss_tensor = -self.discount * q / behavior * pi_tensor[0, action]
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(grad_tensors,
                                                     self.actor_net.variables))

        # 训练critic
        next_q = self.critic_net.predict(next_observation[np.newaxis])[0, action]
        u = reward + self.gamma * (1. - done) * next_q
        u_tensor = tf.convert_to_tensor(u, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q_tensor = self.critic_net(x_tensor)
            mse_tensor = tf.losses.MSE(u_tensor, q_tensor)
            loss_tensor = pi / behavior * mse_tensor
        grad_tensors = tape.gradient(loss_tensor, self.critic_net.variables)
        self.critic_net.optimizer.apply_gradients(zip(grad_tensors,
                                                     self.critic_net.variables))

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma

class RandomAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n

    def decide(self, observation):
        action = np.random.choice(self.action_n)
        behavior = 1. / self.action_n
        return action, behavior


actor_kwargs = {'hidden_sizes': [100, ], 'learning_rate': 0.0005}
critic_kwargs = {'hidden_sizes': [100, ], 'learning_rate': 0.0005}
agent = OffPACAgent(env, actor_kwargs=actor_kwargs,
                    critic_kwargs=critic_kwargs)
behavior_agent = RandomAgent(env)

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
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward

# 训练
episodes = 80
episode_rewards = []
for episode in range(episodes):
    observation = env.reset()
    episode_reward = 0.
    while True:
        action, behavior = behavior_agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.learn(observation, action, behavior, reward,
                    next_observation, done)
        if done:
            break
        observation = next_observation

    # 跟踪监控
    episode_reward = play_qlearning(env, agent)
    episode_rewards.append(episode_reward)

plt.plot(episode_rewards)
plt.show()

# 测试
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                     len(episode_rewards), np.mean(episode_rewards)))

env.close()
