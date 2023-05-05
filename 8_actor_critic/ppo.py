import numpy as np
import pandas as pd
np.random.seed(0)
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow import keras
from pporeplayer import PPOReplayer

env = gym.make('Acrobot-v1')
env.seed(0)

# 临近策略优化算法,离散动作ppo
# lambd:lambda_advantage
class PPOAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, clip_ratio=0.1,
                 gamma=0.99, lambd=0.99, min_trajectory_length=1000,
                 batches=1, batch_size=64):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.lambd = lambd
        self.clip_ratio = clip_ratio
        self.min_trajectory_length = min_trajectory_length
        self.batches = batches
        self.batch_size = batch_size
        self.trajectory = []
        self.replayer = PPOReplayer()
        self.actor_net = self.build_network(output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            **actor_kwargs)
        self.critic_net = self.build_network(output_size=1,
                                             **critic_kwargs)
    # input_size?
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
        # observation[np.newaxis]相比于observation增加了一个维度，变为1*n
        # observation ndarray(6, )   observation[np.newaxis]   ndarray(1, 6)
        probs = self.actor_net.predict(observation[np.newaxis])[0]        # ndarray(3, )
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
            observations = np.stack(df['observation'])
            df['v'] = self.critic_net.predict(observations)
            pis = self.actor_net.predict(observations)
            df['pi'] = [pi[action] for pi, action in zip(pis, df['action'])]

            df['next_v'] = df['v'].shift(-1).fillna(0.)         # 数据向上移动，nan值以0填充
            df['u'] = df['reward'] + self.gamma * df['next_v']
            df['delta'] = df['u'] - df['v']
            df['return'] = df['reward']
            df['advantage'] = df['delta']
            for i in df.index[-2::-1]:
                df.loc[i, 'return'] += self.gamma * df.loc[i + 1, 'return']
                df.loc[i, 'advantage'] += self.gamma * self.lambd *\
                    df.loc[i + 1, 'advantage']
            fields = ['observation', 'action', 'pi', 'advantage', 'return']
            self.replayer.store(df[fields])
            self.trajectory = []

            if len(self.replayer.memory) > self.min_trajectory_length:
                for batch in range(self.batches):
                    observations, actions, pis, advantages, returns =\
                        self.replayer.sample(size=self.batch_size)
                    # 训练actor
                    s_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
                    gather_tensor = tf.convert_to_tensor([(i, a) for i, a
                                    in enumerate(actions)], dtype=tf.int32)
                    pi_old_tensor = tf.convert_to_tensor(pis, dtype=tf.float32)
                    advantage_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)

                    with tf.GradientTape() as tape:
                        all_pi_tensor = self.actor_net(s_tensor)
                        # def gather_nd(params, indices, name=None)
                        # 根据indices描述的索引，提取params上的元素，重新构建一个tensor
                        pi_tensor = tf.gather_nd(all_pi_tensor, gather_tensor)
                        surrogate_advantage_tensor = (pi_tensor / pi_old_tensor) * advantage_tensor
                        clip_times_advantage_tensor = self.clip_ratio * surrogate_advantage_tensor
                        max_surrogate_advantage_tensor = advantage_tensor + \
                            tf.where(advantage_tensor > 0., clip_times_advantage_tensor, -clip_times_advantage_tensor)
                        clipped_surrogate_advantage_tensor = tf.minimum(
                            surrogate_advantage_tensor, max_surrogate_advantage_tensor)
                        loss_tensor = -tf.reduce_mean(clipped_surrogate_advantage_tensor)

                    actor_grads = tape.gradient(loss_tensor, self.actor_net.variables)
                    self.actor_net.optimizer.apply_gradient(zip(actor_grads, self.actor_net.variables))

                    # 训练critic
                    self.critic_net.fit(observations, returns, verbose=0)      # 状态价值函数

                self.replayer = PPOReplayer()

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

actor_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.001}
critic_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.002}
agent = PPOAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs, batches=50)

# 训练
episodes = 200
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_montecarlo(env, agent, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()

# 测试
episode_rewards = [play_montecarlo(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

env.close()