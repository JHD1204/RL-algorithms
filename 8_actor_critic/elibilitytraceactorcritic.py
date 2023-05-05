import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from advantageactorcritic import AdvantageActorCriticAgent

env = gym.make('Acrobot-v1')
env.seed(0)

# 带资格迹的优势Actor-Critic算法
class ElibilityTraceActorCriticAgent(AdvantageActorCriticAgent):
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99,
                 actor_lambda=0.9, critic_lambda=0.9):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.gamma = gamma
        self.discount = 1.

        self.actor_net = self.build_network(input_size=observation_dim, output_size=self.action_n,
                                            output_activation=tf.nn.softmax, **actor_kwargs)
        self.critic_net = self.build_network(input_size=observation_dim, output_size=1., **critic_kwargs)
        self.actor_traces = [np.zeros_like(weight) for weight in self.actor_net.get_weights()]
        self.critic_traces = [np.zeros_like(weight) for weight in self.critic_net.get_weights()]

    def learn(self, observation, action, reward, next_observation, done):
        q = self.critic_net.predict(observation[np.newaxis])[0, 0]
        u = reward + (1. - done) * self.gamma * self.critic_net.predict(next_observation[np.newaxis])[0, 0]
        td_error = u - q
        # 训练actor网络
        x_tensor = tf.convert_to_tensor(observation[np.newaxis], dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            logpi_pick_tensor = logpi_tensor[0, action]
        grad_tensors = tape.gradient(logpi_pick_tensor, self.actor_net.variables)
        self.actor_traces = [self.gamma * self.actor_lambda * trace +
                             self.discount * grad.numpy() for trace, grad in
                             zip(self.actor_traces, grad_tensors)]
        actor_grads = [tf.convert_to_tensor(-td_error * trace, dtype=tf.float32)\
                       for trace in self.actor_traces]
        actor_grads_and_vars = tuple(zip(actor_grads, self.actor_net.variables))
        self.actor_net.optimizer.apply_gradients(actor_grads_and_vars)

        # 训练critic网络
        with tf.GradientTape() as tape:
            v_tensor = self.critic_net(x_tensor)
        grad_tensors = tape.gradient(v_tensor, self.critic_net.variables)
        self.critic_traces = [self.gamma * self.critic_lambda * trace +
                              self.discount * grad.numpy() for trace, grad in
                              zip(self.critic_traces, grad_tensors)]
        critic_grads = [tf.convert_to_tensor(-td_error * trace, dtype=tf.float32)
                        for trace in self.critic_traces]
        critic_grads_and_vars = tuple(zip(critic_grads, self.critic_net.variables))
        self.critic_net.optimizer.apply_gradients(critic_grads_and_vars)

        if done:
            self.actor_traces = [np.zeros_like(weight) for weight in self.actor_net.get_weights()]
            self.critic_traces = [np.zeros_like(weight) for weight in self.critic_net.get_weights()]
            self.discount = 1.
        else:
            self.discount *= self.gamma

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

actor_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.001}
critic_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.001}
agent = ElibilityTraceActorCriticAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)

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