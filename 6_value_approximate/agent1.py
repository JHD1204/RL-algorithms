import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env = env.unwrapped
print('观测空间 = {}'.format(env.observation_space))
observation_dim = env.observation_space.shape[0]
print('观测 = {}'.format(observation_dim))
print('动作空间 = {}'.format(env.action_space))
print('动作 = {}'.format(env.action_space.n))
print('位置范围 = {}'.format((env.min_position, env.max_position)))
print('速度范围 = {}'.format((-env.max_speed, env.max_speed)))
print('目标位置 = {}'.format(env.goal_position))

positions, velocities = [], []
observation = env.reset()
count = 0
while True:
    positions.append(observation[0])
    velocities.append(observation[1])
    next_observation, reward, done, _ = env.step(2)
    if done or (count >= 200):
        break
    observation = next_observation
    print('count = {}'.format(count))
    count += 1

if next_observation[0] > 0.5:
    print('成功到达')
else:
    print('失败退出')

fig, ax = plt.subplots()
ax.plot(positions, label='position')
ax.plot(velocities, label='velocity')
ax.legend()
fig.show()

