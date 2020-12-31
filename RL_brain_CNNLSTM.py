# -*- coding: UTF-8 -*-

"""
这个是DQN大脑部分
使用Keras建造神经网络
使用CNN加LSTM做神经网络
可以看我的CSDN：https://blog.csdn.net/u012465304
"""
# 如果要用tensorflow就把下面这个注释给删掉，接下来我们先用tensorflow
# import os
# os.environ['KERAS_BACKEND']='theano'
import numpy as np
# 按顺序建立的神经网络
from tensorflow.keras.models import Sequential
# dense是全连接层，这里选择你要用的神经网络层参数
from tensorflow.keras.layers import Dense, Activation,Convolution2D, MaxPooling2D, Flatten
# 选择优化器
from tensorflow.keras.optimizers import Adam, RMSprop
# 画图
from tensorflow.keras.utils import plot_model

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            observation_shape,
            learning_rate=0.01,
            reward_decay=0.9,
            epsilon_max=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
            first_layer_neurno=4,
            second_layer_neurno=1,
            go=None,
            epsilon = None,
            weights_path = None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.observation_shape = observation_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max    # 最多是90%通过神经网络选择，10%随机选择
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = epsilon if e_greedy_increment is not None else self.epsilon_max  # e_greedy_increment 通过神经网络选择的概率慢慢增加
        self.first_layer_neurno = first_layer_neurno
        self.second_layer_neurno = second_layer_neurno
        self.go=go
        self.weights_path = weights_path

        # total learning step
        self.learn_step_counter = 0

        # 由于图像数据太大了 分开用numpy存
        # self.memoryList = []
        self.memoryObservationNow = np.zeros((self.memory_size, self.observation_shape[0],
                                              self.observation_shape[1], self.observation_shape[2]), dtype='int16')
        self.memoryObservationLast = np.zeros((self.memory_size, self.observation_shape[0],
                                               self.observation_shape[1], self.observation_shape[2]), dtype='int16')
        self.memoryReward = np.zeros(self.memory_size, dtype='float64')
        self.memoryAction = np.zeros(self.memory_size, dtype='int16')

        # consist of [target_net, evaluate_net]
        self._build_net()

        # print("1")
        if output_graph:
            print("输出图像")
            plot_model(self.model_eval, to_file='model1.png')
            plot_model(self.model_target, to_file='model2.png')

        # 记录cost然后画出来
        self.cost_his = []
        self.reward = []

    def _build_net(self):
        self.model_eval = Sequential([
            # 输入第一层是一个二维卷积层(100, 80, 1)
            Convolution2D(                              # 就是Conv2D层
                batch_input_shape=(None, self.observation_shape[0], self.observation_shape[1],
                                   self.observation_shape[2]),
                filters=15,                             # 多少个滤波器 卷积核的数目（即输出的维度）
                kernel_size=5,                          # 卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
                strides=1,                              # 每次滑动大小
                padding='same',                         # Padding 的方法也就是过滤后数据xy大小是否和之前的一样
                #data_format='channels_last',           # 表示图像通道维的位置，这里rgb图像是最后一维表示通道
            ),
            Activation('relu'),
            # 输出(100, 80, 15)
            # Pooling layer 1 (max pooling) output shape (50, 40, 15)
            MaxPooling2D(
                pool_size=2,                            # 池化窗口大小
                strides=2,                              # 下采样因子
                padding='same',                         # Padding method
                #data_format='channels_last',
            ),
            # output(50, 40, 30)
            Convolution2D(30, 5, strides=1, padding='same',
                          #data_format='channels_last'
            ),
            Activation('relu'),
            # (10, 8, 30)
            MaxPooling2D(5, 5, 'same',
                         #data_format='channels_first'
                         ),
            # (10, 8, 30)
            Flatten(),
            # LSTM(
            #     units=1024,
            #     return_sequences=True,  # True: output at all steps. False: output as last step.
            #     stateful=True,          # True: the final state of batch1 is feed into the initial state of batch2
            # ),
            Dense(512),
            Activation('relu'),
            Dense(self.n_actions),
        ])

        if self.go == 1:
            pass
        else:
            self.model_eval.load_weights(self.weights_path)

        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model_eval.compile(loss='mse',
                            optimizer=rmsprop,
                            metrics=['accuracy'])

        self.model_target = Sequential([
            Convolution2D(  # 就是Conv2D层
                batch_input_shape=(None, self.observation_shape[0], self.observation_shape[1],
                                   self.observation_shape[2]),
                filters=15,  # 多少个滤波器 卷积核的数目（即输出的维度）
                kernel_size=5,  # 卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
                strides=1,  # 每次滑动大小
                padding='same',  # Padding 的方法也就是过滤后数据xy大小是否和之前的一样
                #data_format='channels_last',  # 表示图像通道维的位置，这里rgb图像是最后一维表示通道
            ),
            Activation('relu'),
            # 输出（210， 160， 30）
            # Pooling layer 1 (max pooling) output shape (105, 80, 30)
            MaxPooling2D(
                pool_size=2,  # 池化窗口大小
                strides=2,  # 下采样因子
                padding='same',  # Padding method
                #data_format='channels_last',
            ),
            # output(105, 80, 60)
            Convolution2D(30, 5, strides=1, padding='same',
                          #data_format='channels_last'
                          ),
            Activation('relu'),
            # (21, 16, 60)
            MaxPooling2D(5, 5, 'same',
                         #data_format='channels_first'
                         ),
            # 21 * 16 * 60 = 20160
            Flatten(),
            # LSTM(
            #     units=1024,
            #     return_sequences=True,  # True: output at all steps. False: output as last step.
            #     stateful=True,          # True: the final state of batch1 is feed into the initial state of batch2
            # ),
            Dense(512),
            Activation('relu'),
            Dense(self.n_actions),
        ])

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        s = s[:, :, np.newaxis]
        s_ = s_[:, :, np.newaxis]
        # print(s.shape())
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memoryObservationNow[index, :] = s_
        self.memoryObservationLast[index, :] = s
        self.memoryReward[index] = r
        self.memoryAction[index] = a

        self.memory_counter += 1

    def choose_action(self, observation):
        # 插入一个新的维度 矩阵运算时需要新的维度来运算
        observation = observation[np.newaxis, :, :, np.newaxis]


        if np.random.uniform() < self.epsilon:
            # 向前反馈，得到每一个当前状态每一个action的Q值
            # 这里使用估计网络，也就是要更新参数的网络
            # 然后选择最大值,这里的action是需要执行的action
            # print(observation)
            actions_value = self.model_eval.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
            # print(action)
        return action

    def learn(self):
        # 经过一定的步数来做参数替换
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
            print('\ntarget_params_replaced\n')

        # 随机取出记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memoryONow = self.memoryObservationNow[sample_index, :]
        batch_memoryOLast = self.memoryObservationLast[sample_index, :]
        batch_memoryAction = self.memoryAction[sample_index]
        batch_memoryReward = self.memoryReward[sample_index]


        # 这里需要得到估计值加上奖励 成为训练中损失函数的期望值
        # q_next是目标神经网络的q值，q_eval是估计神经网络的q值
        # q_next是用现在状态得到的q值 q_eval是用这一步之前状态得到的q值
        # print(batch_memory[:, -self.n_features:])
        q_next = self.model_target.predict(batch_memoryONow, batch_size=self.batch_size)
        q_eval = self.model_eval.predict(batch_memoryOLast, batch_size=self.batch_size)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memoryAction.astype(int)
        reward = batch_memoryReward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.cost = self.model_eval.train_on_batch(batch_memoryONow, q_target)

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        # print(self.epsilon)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



