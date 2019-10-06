# ==============================================================================
# Mountain car
#
# Environmental return:
#   Observations: [location, velocity] (location is negative for left of reward)
#   Reward:
#   Done: whether episode terminated
#   Info: ?? empty dictionary
#
# Gym page:
# Docs on environment:
#
# ==============================================================================

"""SUBSEQUENT To-Do'

- Figure out what the correct feature representation should be
    - should I just "grid"-ize the world?
    - with hand-crafted features it does quite well in fact (see below)
- Figure out what the correct learning algorithm should be
    - perhaps linear learning with discretized world is okay given enough samples?
- Figure out how to derive eligibility traces to a learning objective (i.e. so
    I can modify the feature & results, instead of the intermediate update steps)
- Tune parameters (learning, elibility, etc.)

- have some logging events to save the training epochs
"""

import sys, random, math
import numpy as np
import gym

from sklearn import linear_model
from sklearn.svm import SVR

#Construct the pole-balancing environment
env = gym.make('MountainCar-v0')

#How many episodes
N_EPS = 1000
#How many timesteps each episode?
N_TIMESTEP = 400 # should be 200 - 1000, auto-terminate after 200 usually

#
ACTION_SPACE = np.array([0,1,2]) # TODO switch to using env.action_space instead?


## Agent tuning variables
GAMMA = 0.8 # discount factor for reward over time
LAMBDA = 0.0 # discount factor for eligibiltiy traces
ALPHA = 0.1 # learning rate
INIT_EPSILON = 0.2 # e-greedy policy randomness
EPSILON_DECAY = 1.0


# Dimensionality of the state-action feature
SA_DIM = 8


### Agent ###
class Agent:

    # Constructor to initialize attributes
    def __init__(self):
        # Hyper-parameters
        self.Epsilon = INIT_EPSILON
        # Feature and eligibility vectors
        self.prev_xVec = np.random.rand(SA_DIM)
        self.EligVec = np.empty(SA_DIM)
        # Q-estimation
        self.Q_estimate = linear_model.SGDRegressor(max_iter=1, learning_rate='constant')
        #self.Q_estimate = SVR(max_iter=1, gamma='auto')
        self.Q_estimate.fit(self.prev_xVec[np.newaxis,:], [0])


    # Extract state-action features based on observations
    def state_action_features(self, obs, action):
        action_onehot = np.zeros(3)
        action_onehot[action] = 1.0

        # hand-crafted features
        acc_right = (obs[1] > 0) * (action==2)
        acc_left = (obs[1] < 0) * (action==0)

        sa_Vec = np.concatenate((obs, action_onehot, [acc_right, acc_left], [1.0])) # TODO make more sophisticated

        return sa_Vec


    # Get action from policy
    def e_greedy_policy(self, cur_states):
        # Pick random action
        chosen_action = np.random.choice(ACTION_SPACE, size=None, replace=True, p=None)

        # Chance of greedy action
        if np.random.uniform() > self.Epsilon:
            highest_q = -100.0

            for cur_act in ACTION_SPACE:
                cur_x = self.state_action_features(cur_states, cur_act)
                cur_q_esti = self.Q_estimate.predict(cur_x[np.newaxis,:])


                if cur_q_esti > highest_q:
                    chosen_action = cur_act
                    highest_q = cur_q_esti

        return chosen_action


    # How agent acts
    def act(self, observations, reward):

        # Compute next step action based on current policy
        next_A = self.e_greedy_policy(observations)

        # Compute the state-action vector
        cur_xVec = self.state_action_features(observations, next_A)

        # Compute the boostraped TD target given above action
        td_target = reward + ( GAMMA * self.Q_estimate.predict(cur_xVec[np.newaxis,:]) )

        # Update the Q model
        #self.Q_estimate.fit(self.prev_xVec[np.newaxis,:], td_target)
        self.Q_estimate.partial_fit(self.prev_xVec[np.newaxis,:], td_target)

        # Increment eligibility for state-action NOT SURE HOW TO DERIVE THIS YET
        #EligVec = (GAMMA * LAMBDA * self.EligVec) + cur_xVec

        # (maybe TODO) re-update policy and re-sample action??

        # Save previous states
        self.prev_xVec = cur_xVec

        # Decay exploration
        self.Epsilon = self.Epsilon * EPSILON_DECAY


        return next_A




### Simulate function ###
def simulate(env):

    # Initialize agent
    agent = Agent()

    for i_episode in range(N_EPS):
        print("Episode %d" % i_episode)
        # Variable to count the reward

        # Get environment
        observation = env.reset()

        next_action = agent.act(observation, 0) #TODO potentially change


        for t in range(N_TIMESTEP):
            # Visual render
            env.render()

            # Compute states for next timestep
            (observation, reward, done, info) = env.step(next_action)

            # Compute subsequent action
            #next_action = agent.act(observation, reward)
            aug_reward = observation[0] + reward
            next_action = agent.act(observation, aug_reward)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


        #print("Reward: %d" % rewardAgg)
        print(agent.Q_estimate.coef_)
        #print(agent.Q_estimate.dual_coef_)

    env.close()

### Main loop ###
def main():
    simulate(env)
    print("Done")

main()
