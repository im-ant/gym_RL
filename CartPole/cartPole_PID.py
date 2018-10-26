# ==============================================================================
# Gym page: https://gym.openai.com/envs/CartPole-v1/#barto83
# Tutorial: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# Docs on environment: https://github.com/openai/gym/wiki/CartPole-v0
# Additional resource on tutorial:
#   https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe
#
# ==============================================================================
import sys, random, math
import numpy as np
import gym

#Construct the pole-balancing environment
env = gym.make('CartPole-v0')

#How many episodes
N_EPS = 1
#How many timesteps each episode?
N_TIMESTEP = 1000



# Compute how much cart velocity we should have
def PD_control(theta, dTheta):
    k_P = 10.0
    k_D = 8.0

    wantedCarV = (k_P * theta) + (k_D * dTheta)

    return wantedCarV


# Select the optimal action from the q table
def getAction(curV, goalV):
    if (goalV > curV):
        return 1

    if (goalV < curV):
        return 0

    if (goalV == curV):
        return None


# Simulate function
def simulate(env):

    for i_episode in range(N_EPS):
        print("Episode %d" % i_episode)
        # Variable to count the reward
        rewardAgg = 0

        observation = env.reset()

        for t in range(N_TIMESTEP):
            env.render()
            # Unpack the observations
            carPos, carV, theta, thetaV = observation

            # Compute the
            goal_carV = PD_control(theta, thetaV)


            #Select action
            action = getAction(carV, goal_carV)
            if action == None:
                continue

            #Compute next timestep
            observation, reward, done, info = env.step(action)

            rewardAgg += reward # Count the rewards


            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        print("Reward: %d" % rewardAgg)

    env.close()


def main():
    simulate(env)
    print("Done")

main()
