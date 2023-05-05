import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from dqnreplayer import DQNReplayer
from tensorflow import keras

np.random.seed(0)
env = gym.make('Acrobot-v1')
env.seed(0)

class SACAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs,
                 gamma=0.99, alpha=0.2, net_learning_rate=0.1,
                 replayer_capacity=1000, batches=1, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.net_learning_rate = net_learning_rate

        self.batches = batches
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)

        def sac_loss(y_true, y_pred):
            qs = alpha * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
            return tf.reduce_sum(qs, axis=-1)

        self.actor_net = self.build_network(input_size=observation_dim,
                                            output_size=self.action_n, output_activation=tf.nn.softmax,
                                            loss=sac_loss, **actor_kwargs)
        self.q0_net = self.build_network(input_size=observation_dim,
                                         output_size=self.action_n, **critic_kwargs)
        self.q1_net = self.build_network(input_size=observation_dim,
                                         output_size=self.action_n, **critic_kwargs)
        self.v_evaluate_net = self.build_network(input_size=observation_dim,
                                                 output_size=1, **critic_kwargs)
        self.v_target_net = self.build_network(input_size=observation_dim,
                                               output_size=1, **critic_kwargs)
        self.update_target_net(self.v_target_net, self.v_evaluate_net)

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

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                           for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)

        for batch in range(self.batches):
            observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size)
            pis = self.actor_net.predict(observations)
            q0s = self.q0_net.predict(observations)
            q1s = self.q1_net.predict(observations)

            # 训练actor
            self.actor_net.fit(observations, q0s, verbose=0)

            # 训练critic
            q01s = np.minimum(q0s, q1s)
            entropic_q01s = pis * q01s - self.alpha * scipy.special.xlogy(pis, pis)
            v_targets = entropic_q01s.sum(axis=-1)
            self.v_evaluate_net.fit(observations, v_targets, verbose=0)

            next_vs = self.v_target_net.predict(next_observations)
            q_targets = rewards + self.gamma * (1. - dones) * next_vs[:, 0]
            q0s[range(self.batch_size), actions] = q_targets
            q1s[range(self.batch_size), actions] = q_targets
            self.q0_net.fit(observations, q0s, verbose=0)
            self.q1_net.fit(observations, q1s, verbose=0)

actor_kwargs = {'hidden_sizes':[100, ], 'learning_rate':0.01}
critic_kwargs = {'hidden_sizes':[100, ], 'learning_rate':0.01}
agent = SACAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs, batches=5)

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
            # 更新目标网络
            agent.update_target_net(agent.v_target_net, agent.v_evaluate_net, agent.net_learning_rate)
            break
        observation = next_observation
    return episode_reward

# 训练
episodes = 100
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
    print('episode {} game over'.format(episode))
plt.plot(episode_rewards)
plt.show()

# 测试
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))