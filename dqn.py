# -*- coding: utf-8 -*-
import random
from collections import deque

import gym
import keras
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

import JobShop

EPISODES = 10000


class DQNAgent:
    def __init__(self, state_size, action_size, number_job, number_feature):
        self.state_size = state_size
        self.action_size = action_size
        self.number_job = number_job
        self.number_feature = number_feature
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        basic_model = self._basic_model()
        # basic_model = Dense(10)

        output_list = []
        input_list = []
        for i in range(self.number_job):
            input_list.append(Input(shape=(self.number_feature,)))
            output_list.append(basic_model(input_list[i]))

        concatenated = keras.layers.concatenate(output_list)
        out = Dense(self.action_size, activation='linear')(concatenated)
        model = Model(input_list, out)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _basic_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential(name='basic_model')
        model.add(Dense(24, input_dim=self.number_feature, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    # def _build_model(self):
    #     # Neural Net for Deep-Q learning Model
    #     model = Sequential() 
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss='mse',
    #                   optimizer=Adam(lr=self.learning_rate))
    #     return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # input(act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax( self.model.predict(  next_state )[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# if __name__ == "__main__":
#     env = gym.make('CartPole-v1')
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)
#     # agent.load("./save/cartpole-dqn.h5")
#     done = False
#     batch_size = 32

#     for e in range(EPISODES):
#         state = env.reset()
#         state = np.reshape(state, [1, state_size])
#         for time in range(500):
#             # env.render()
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             reward = reward if not done else -10
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}"
#                       .format(e, EPISODES, time, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)
#         # if e % 10 == 0:
#         #     agent.save("./save/cartpole-dqn.h5")

if __name__ == "__main__":
    number_job = 5
    number_machine = 4
    number_feature = 1

    # env = gym.make('CartPole-v1')
    state_size = number_job * number_feature
    action_size = number_job
    agent = DQNAgent(state_size, action_size, number_job, number_feature)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = number_job * number_machine * 10

    history = []

    for e in range(EPISODES):
        problem = JobShop.JobShop(number_machine, number_job, 15, 30)
        state, score, done = problem.Step()
        state = np.reshape(state, [1,state_size,1])
        state = [state[0][i] for i in range(number_job)]
        score = 0
        oldscore = 0

        action_list = []
        for time in range(number_job*number_machine):
            # env.render()
            # input(state)
            action = agent.act(state)
            # input(action)
            action_list.append(action)
            next_state, score, done = problem.Step(action)
            reward = oldscore - score + 15 if not done else -1000
            oldscore = score
            next_state = np.reshape(next_state, [1,state_size,1])
            next_state = [next_state[0][i] for i in range(number_job)]
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # input(reward)
            # input('-------')
            if done:
                print("episode: {}/{}, loop: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break

        if len(agent.memory) > batch_size:
            print('replay')
            agent.replay(batch_size)
        
        
        # problem.PlotResult() 
        history.append(score)
        if e % 10 == 0:
            meanscore = np.mean( score )
            print(e, meanscore, agent.epsilon)
            print(action_list)
            f = open('his', 'a')
            f.write(str(meanscore)+'\n')
            f.close()

        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

    # problem = JobShop.JobShop(4, 5, 15, 30)
    # po = problem.Get_Possible_Job_Position()
    # problem.MeasurementAction(problem.schedule_line)
    # problem.Get_Features(po)
    # print(a)
