"""
Deep Q network,

"""


import gym
from RL_brain_CNNLSTM import DeepQNetwork
import cv2
import numpy as np
import time
from receiveThread import myThread

env = gym.make('SpaceInvaders-v0')
env = env.unwrapped

'''
print(env.action_space)
# print(env.observation_space)
print(env.observation_space.shape)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.reward_range)
'''

inputImageSize = (100, 80, 1)
# inputImageSize[2] = 1

print("从头：1||继续：2")
go = input("请选择：")
go = int(go)
if go==1:
    with open('log.txt','w') as f:
        f.write('')
        f.close()
    with open('reward.txt','w') as f:
        f.write('')
        f.close()
    epsilon = 0
    weights_path = None
elif go == 2:
    with open('epsilon.txt','r') as f:
        epsilon = f.readline()
        f.close()
    epsilon = float(epsilon)
    weights_path = 'weights\\eval_weights'
else:
    exit()
RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  observation_shape=inputImageSize,
                  learning_rate=0.01, epsilon_max=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.0001,
                  output_graph=True,
                  go = go,
                  epsilon = epsilon,
                  weights_path = weights_path)

total_steps = 0

total_reward_list = []
for i_episode in range(1000000):

    observation = env.reset()
    # 使用opencv做灰度化处理
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (inputImageSize[1], inputImageSize[0]))
    total_reward = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        reward = reward / 200

        observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
        observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))

        RL.store_transition(observation, action, reward, observation_)

        total_reward += reward
        if total_steps > 1024 and total_steps % 8 == 0:
            t0 = time.time()
            RL.learn()
            t1 = time.time()

        if done:
            total_reward_list.append(total_reward)
            print('episode: ', i_episode,
                  'total_reward: ', round(total_reward, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            # plot_reward()
            # print('total reward list:', total_reward_list)
            break

        observation = observation_
        total_steps += 1
    if i_episode%5==0:
        with open('log.txt','a') as f:
            for item in RL.cost_his:
                f.write(str(item)+'\n')
            f.close()
            RL.cost_his = []
        with open('reward.txt', 'a') as f:
            for item in total_reward_list:
                f.write(str(item) + '\n')
            f.close()
            total_reward_list = []
        with open('epsilon.txt', 'w') as f:
            f.write(str(RL.epsilon))
            f.close()
        RL.model_eval.save_weights('weights\\eval_weights', save_format='tf')

RL.plot_cost()


