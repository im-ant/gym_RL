# ==============================================================================
# Un-actuated pole balancing on cart, with control being force applied to cart
#
# Gym page: https://gym.openai.com/envs/CartPole-v1/#barto83
# Docs on environment: https://github.com/openai/gym/wiki/CartPole-v0
# ==============================================================================
import sys, random, math
#import numpy as np
import gym

#Construct the pole-balancing environment
env = gym.make('CartPole-v0')

#How many episodes
N_EPS = 1
#How many timesteps each episode?
N_TIMESTEP = 1000


# Use PD control to find the desired cart velocity given angle
def PD_control(theta, dTheta):
    # Constants
    k_P = 10.0
    k_D = 8.0

    wantedCarV = (k_P * theta) + (k_D * dTheta)

    return wantedCarV


# Bang-bang control based on current and desired cart velocity
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

            # Compute the desired cart velocity
            goal_carV = PD_control(theta, thetaV)

            # Select action
            action = getAction(carV, goal_carV)
            if action == None:
                continue

            # Compute states for next timestep
            observation, reward, done, info = env.step(action)

            # Count rewards
            rewardAgg += reward # Count the rewards


            if done:
                continue
                print("Episode finished after {} timesteps".format(t+1))
                break

        print("Reward: %d" % rewardAgg)

    env.close()


def main():
    simulate(env)
    print("Done")

main()
