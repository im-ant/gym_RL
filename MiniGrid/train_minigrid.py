# =============================================================================
# Training the agent in classical control environments from visual input
#
#
# Potential Resources:
# - Some baseline plots from Google Dopamine:
#   https://google.github.io/dopamine/baselines/plots.html
# - A discussion on (the lack of) frame maxing:
#   https://github.com/openai/gym/issues/275
# - The DQN hyper-parameters, as reported by Google Dopamine:
#   https://github.com/google/dopamine/tree/master/dopamine/agents/dqn/configs
# - Saving images:
#   save_image(torch.tensor(tmp_obs, dtype=float).unsqueeze(0),
#                f'tmp_img{args.env_name}.png', normalize=True, range=(0, 255))
#
#
# Author: Anthony G. Chen
# =============================================================================

import argparse
import math
import sys

import gym
from gym_minigrid import wrappers

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import dqn_agent
from env_wrapper import MiniGridImgWrapper


def _init_agent(args: argparse.Namespace, env, device):
    """
    Method for initializing the agent
    :param args: argparse arguments
    :return: agent instance
    """

    # TODO: need to add num_actons and observation_shape

    # =====================================================
    # Initialize agent
    agent = dqn_agent.DQNAgent(num_actions=env.action_space.n,
                               observation_shape=env.observation_space.shape,
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
                               device=device)

    # TODO: implement memory buffer location?

    return agent


def run_environment(args: argparse.Namespace,
                    device: str = 'cpu',
                    logger=None):
    # ==============================
    # Initialize environment and pre-processing

    # Whether to use pixel space or the compact encoding
    use_pixel_space = False
    # If pixel space, side length of the observable square (in # tiles)
    # the pixel image has shape (3, 7*tile_size, 7*tile_size)
    tile_size = 8

    # Initialize environment and set it to image space
    env = gym.make(args.env_name)
    if use_pixel_space:
        env = wrappers.RGBImgPartialObsWrapper(env, tile_size=tile_size)
    env = wrappers.ImgObsWrapper(env)
    env = MiniGridImgWrapper(env)  # my wrapper

    # NOTE TODO: eventual goal is to have all of OpenAI Gym under the same training procedure

    """
    # TODO organize reproducibility settings
    raw_env.seed(args.seed)  # reproducibility settings
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    """

    # ==============================
    # Initialize agent
    agent = _init_agent(args, env, device)

    # =====================================================
    # Start interacting with environment
    for episode_idx in range(args.num_episode):

        # Reset counters
        cumulative_reward = 0.0
        steps = 0

        # Reset environment and agent
        observation = env.reset()
        action = agent.begin_episode(observation)

        while True:
            # Interact
            observation, reward, done, info = env.step(action)
            action = agent.step(observation, reward, done)

            # Update variables
            cumulative_reward += reward
            steps += 1

            if done:
                # =========================================
                # Logging stuff

                if episode_idx == 0:
                    print("Epis || Steps || Return")

                if logger is None:
                    print(f"{episode_idx} || {steps} || {cumulative_reward}")
                else:
                    logger.add_scalar('Reward', cumulative_reward,
                                      global_step=episode_idx)
                    if episode_idx % 10 == 0:
                        print(f"{episode_idx} || {steps} || {cumulative_reward}")

                agent.report(episode_idx, logger=logger)

                """
                Old log for reference
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
                """

                # =========================================
                # Break out of current episode
                break


if __name__ == "__main__":

    # TODO: have a hyperparmeter .config file for the future

    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='DQN for atari environment')

    # Environmental parameters
    # TODO change and test all the parameters here for control
    # TODO check that having 1 frame skip isn't going to break things (useful for grid world)
    parser.add_argument('--env_name', type=str,
                        default='MiniGrid-Empty-6x6-v0', metavar='N',
                        help='environment to initialize (default: CartPole-v1')
    parser.add_argument('--num_episode', type=int, default=20, metavar='N',
                        help='number of episodes to run the environment for (default: 500)')
    parser.add_argument('--frame_skips', type=int, default=1, metavar='N',
                        help="""number of frames to repeat each action for, the agent does
                                    not observe the in-between frames. Note that if set to 1 the
                                    AtariPreprocessing max pooling breaks down(default: 4)""")
    # TODO: add a functionality of max-pooling k previous frames?

    # Agent parameters
    parser.add_argument('--history_size', type=int, default=2, metavar='N',
                        help='number of most recent observations to construct a state (default: 4)')
    # NOTE: no action-repeat hyperparameter, since Rl-gym automatically do action repeat (sampled between 2-4)
    parser.add_argument('--update_period', type=int, default=4, metavar='N',
                        help='num of actions selected between SGD updates (default: 4)')
    parser.add_argument('--target_update-period', type=int, default=16, metavar='N',
                        help="""frequency to update target network, as measured in the number of actions
                                NOTE actually it is the number of optimization steps now, change? TODO
                                (default: 8000)""")

    parser.add_argument('--buffer_capacity', type=int, default=20000, metavar='N',
                        help='capacity of the non-parametric replay buffer (default: 1,000,000)')

    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='g',
                        help='discount factor (gamma) for future reward (default: 0.99)')
    parser.add_argument('--minibatch_size', type=int, default=256, metavar='N',
                        help='batch size for SGD training update (default: 32)')

    parser.add_argument('--init_exploration', type=float, default=1.0, metavar='N',
                        help='initial e-greedy exploration value (default: 1.0)')
    parser.add_argument('--final_exploration', type=float, default=0.05, metavar='N',
                        help='final e-greedy exploration value (default: 0.1)')
    parser.add_argument('--eps_decay_duration', type=int, default=100000, metavar='N',
                        help="""number of actions over which the initial exploration rate is linearly
                                annealed to the final exploration rate (default: 250,000)""")
    parser.add_argument('--min_replay_history', type=int, default=1024, metavar='N',
                        help="""number of transitions / actions to experience (with random 
                                action) before replay learning starts (default: 20,000)""")

    # Experimental parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_dir', type=str, default=None,
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
    if args.log_dir is not None:
        # Tensorboard logger
        logger = SummaryWriter(log_dir=args.log_dir)
        # Add hyperparameters
        logger.add_hparams(hparam_dict=vars(args), metric_dict={})
    else:
        logger = None

    # =====================================================
    # Start environmental interactions
    run_environment(args, device=device, logger=logger)
