import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras

env = gym.make('Acrobot-v1')
env.seed(0)

# 优势Actor-Critic算法
class AdvantageActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.
        self.actor_net = self.build_network(output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            loss=tf.losses.categorical_crossentropy,
                                            **actor_kwargs)
        self.critic_net = self.build_network(output_size=1,
                                             **critic_kwargs)

    def build_network(self, hidden_sizes, output_size, input_size=None,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        x = observation[np.newaxis]
        u = reward + (1. - done) * self.gamma *\
            self.critic_net.predict(next_observation[np.newaxis])
        td_error = u - self.critic_net.predict(x)
        # 训练actor网络
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            loss_tensor = -self.discount * td_error * logpi_tensor
            grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
            self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))
        # 训练critic网络
        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma

actor_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.0001}
critic_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.0002}
agent = AdvantageActorCriticAgent(env, actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs)

def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    step = 0
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
        step += 1
        observation = next_observation
    return episode_reward

# 训练
episodes = 100
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()

# 测试
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

env.close()