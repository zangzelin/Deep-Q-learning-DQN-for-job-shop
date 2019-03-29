# -*- coding: utf-8 -*-
import random
from collections import deque

import keras
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

import JobShop

EPISODES = 10000


class DQNAgent:
    # class for deep q learning agent 
    
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
        self.learning_rate = 0.0005
        self.model = self._build_subproblem_model() # build the model 

    def _build_subproblem_model(self):
        # to build the whole model for jobshop  

        basic_model = self._submodel()

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

    def _submodel(self):
        # the sub model called by function  _build_subproblem_model

        model = Sequential(name='basic_model')
        model.add(Dense(24, input_dim=self.number_feature, activation='relu'))
        model.add(Dense(24, input_dim=self.number_feature, activation='relu'))
        model.add(Dense(24, input_dim=self.number_feature, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _easymodel(self):
        # the easy ann model, not used in this method 
        
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # remember the information of this step

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # let the agent make a decision
        # choose a job to process in current state

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # replay the history and train the model

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # load the model
        self.model.load_weights(name)

    def save(self, name):
        # save the model
        self.model.save_weights(name)


if __name__ == "__main__":

    # agent.load("./save/jobshop-dqn.h5")
    number_job = 5
    number_machine = 4
    number_feature = 2
    state_size = number_job * number_feature
    action_size = number_job
    agent = DQNAgent(state_size, action_size, number_job, number_feature)
    batch_size = number_job * number_machine * 10

    history = []
    successnumber = 0

    # the main loop for each job shop problem 
    for e in range(EPISODES):
        
        problem = JobShop.JobShop(number_machine, number_job, 15, 30, False)
        state, score, done = problem.Step()
        action_list = []
        oldscore = 0
        score = 0

        # the sub loop for each step of the problem 
        for time in range(number_job*number_machine):

            action = agent.act(state)
            next_state, score, done = problem.Step(action)
            reward = oldscore - score + 15 if not done else -1000
            oldscore = score
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                if time >= number_job * number_machine-1:
                    successnumber += 1
                break

            # record the history 
            action_list.append(action)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


        # problem.PlotResult()
        if e % 10 == 0:
            print("loop : {}/{},  score: {} success: {} / 10, e: {:.2}"
                  .format(e, EPISODES, score, successnumber, agent.epsilon))
            print(action_list, len(action_list))
            f = open('log/logs', 'a')
            f.close()
            successnumber = 0

            agent.save("./save/jobshop-dqn.h5")
