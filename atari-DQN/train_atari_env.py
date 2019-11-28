# =============================================================================
# Training the agent in an atari environment
#
#
# Potential Resources:
# - Some baseline plots from Google Dopamine:
#   https://google.github.io/dopamine/baselines/plots.html
# - A discussion on (the lack of) frame maxing:
#   https://github.com/openai/gym/issues/275
# Author: Anthony G. Chen
# =============================================================================

import argparse
from collections import namedtuple
import logging
import math
import random
import sys

import gym
import numpy as np
import torch

import atari_lib
import dqn_agent

# Define things to log
LogTupStruct = namedtuple('Log', field_names=['episode_idx',
                                              'steps',
                                              'buffer_size',
                                              'training_steps',
                                              'returns',
                                              'policy_net_loss'
                                              ])


def init_logger(logging_path: str) -> logging.Logger:
    """
    Initializes the path to write the log to
    :param logging_path: path to write the log to
    :return: logging.Logger object
    """
    logger = logging.getLogger('Experiment-Log')  # not need to change, just names the var instance

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logging_path)
    formatter = logging.Formatter('%(asctime)s||%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_environment(args: argparse.Namespace, device: str = 'cpu',
                    logger=None):

    # =====================================================
    # Initialize environment and pre-processing

    screen_size = 84
    raw_env = gym.make(args.env_name)

    raw_env.seed(args.seed)  # reproducibility settings
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    environment = atari_lib.AtariPreprocessing(raw_env,
                                               frame_skip=args.frame_skips,
                                               terminal_on_life_loss=False,
                                               screen_size=screen_size)

    num_actions = environment.action_space().n
    observation_shape = (1, screen_size, screen_size)

    # =====================================================
    # Initialize agent
    agent = dqn_agent.DQNAgent(num_actions=num_actions,
                               observation_shape=observation_shape,
                               observation_dtype=torch.uint8,
                               history_size=args.history_size,
                               gamma=args.discount_factor,
                               min_replay_history=args.min_replay_history,
                               update_period=args.update_period,
                               target_update_period=args.target_update_period,
                               epsilon_start=args.init_exploration,
                               epsilon_final=args.final_exploration,
                               epsilon_decay_period=args.eps_decay_duration,
                               memory_buffer_capacity=args.buffer_capacity,
                               minibatch_size=args.minibatch_size,
                               device=device,
                               summary_writer=None)  # TODO implement summary writer

    # =====================================================
    # Start interacting with environment
    for episode_idx in range(args.num_episode):

        agent.begin_episode()
        observation = environment.reset()

        cumulative_reward = 0.0
        steps = 0

        while True:

            action = agent.step(observation)
            observation, reward, done, info = environment.step(action)
            # TODO: see if reward is clipped

            agent.store_transition(action, observation, reward, done)

            # Tracker variables
            cumulative_reward += reward
            steps += 1

            if done:
                # =========================================
                # Logging stuff
                # Compute logging variables
                avg_policy_net_loss = 0.0
                if agent.episode_total_policy_loss > 0.0:
                    avg_policy_net_loss = agent.episode_total_policy_loss / \
                                          agent.episode_total_optim_steps
                # TODO: might be nice to compute the final epsilon per episode

                logtuple = LogTupStruct(episode_idx=episode_idx, steps=steps,
                                        buffer_size=len(agent.memory),
                                        training_steps=agent.training_steps,
                                        returns=cumulative_reward,
                                        policy_net_loss=avg_policy_net_loss)

                # Write log
                log_str = '||'.join([str(e) for e in logtuple])
                if args.log_path is not None:
                    logger.info(log_str)
                else:
                    print(log_str)

                # # =========================================
                # Break out of current episode
                break


if __name__ == "__main__":
    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='DQN for atari environment')

    # Environmental parameters
    parser.add_argument('--env-name', type=str, default='MsPacman-v0', metavar='N',
                        help='environment to initialize (default: MsPacman-v0)')
    parser.add_argument('--num-episode', type=int, default=500, metavar='N',
                        help='number of episodes to run the environment for (default: 500)')
    parser.add_argument('--frame-skips', type=int, default=1, metavar='N',
                        help="""number of frames to repeat each action for, the agent does
                                not observe the in-between frames (default: 1)""")
    # TODO: add a functionality of max-pooling k previous frames?

    # Agent parameters
    parser.add_argument('--history-size', type=int, default=4, metavar='N',
                        help='number of most recent observations to construct a state (default: 4)')
    # NOTE: no action-repeat hyperparameter, since Rl-gym automatically do action repeat (sampled between 2-4)
    parser.add_argument('--update-period', type=int, default=4, metavar='N',
                        help='num of actions selected between SGD updates (default: 4)')
    parser.add_argument('--target-update-period', type=int, default=8000, metavar='N',
                        help="""frequency to update target network, as measured in the number of actions
                                (default: 8000)""")

    parser.add_argument('--buffer-capacity', type=int, default=1000000, metavar='N',
                        help='capacity of the non-parametric replay buffer (default: 1,000,000)')

    parser.add_argument('--discount-factor', type=float, default=0.99, metavar='g',
                        help='discount factor (gamma) for future reward (default: 0.99)')
    parser.add_argument('--minibatch-size', type=int, default=32, metavar='N',
                        help='batch size for SGD training update (default: 32)')

    parser.add_argument('--init-exploration', type=float, default=1.0, metavar='N',
                        help='initial e-greedy exploration value (default: 1.0)')
    parser.add_argument('--final-exploration', type=float, default=0.1, metavar='N',
                        help='final e-greedy exploration value (default: 0.1)')
    parser.add_argument('--eps-decay-duration', type=int, default=250000, metavar='N',
                        help="""number of actions over which the initial exploration rate is linearly
                                annealed to the final exploration rate (default: 250,000)""")
    parser.add_argument('--min-replay-history', type=int, default=20000, metavar='N',
                        help="""number of transitions / actions to experience (with random 
                                action) before replay learning starts (default: 20,000)""")

    # Experimental parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-path', type=str, default=None,
                        help='file path to the log file (default: None, printout instead)')
    parser.add_argument('--tmpdir', type=str, default='./',
                        help='temporary directory to store dataset for training (default: cwd)')

    args = parser.parse_args()
    print(args)

    # =====================================================
    # Initialize GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # =====================================================
    # Initialize logging
    log_title_str = '||'.join(LogTupStruct._fields)
    if args.log_path is not None:
        logger = init_logger(args.log_path)
        logger.info(log_title_str)
    else:
        print(log_title_str)
        logger = None

    # =====================================================
    # Start environmental interactions
    run_environment(args, device=device, logger=logger)

