# =============================================================================
# The DQN agent
#
# Very much inspired by / adopted from the Dopamine library:
# https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py
#
#
# Author: Anthony G. Chen
# =============================================================================

from collections import deque
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim

import network
import replay_memory


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon_final):
    """
    TODO: organize stuff here and add start epsilon argument

    :param decay_period: float, the period over which epsilon is decayed.
    :param step: int, the number of training steps completed so far.
    :param warmup_steps:
    :param epsilon:
    :return:
    """
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
    decay_period: 
    step: 
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon_final: float, the final value to which to decay the epsilon parameter.
    Returns:
    A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon_final) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon_final)
    return epsilon + bonus


class DQNAgent(object):
    """
    The DQN Agent
    """

    def __init__(self, num_actions: int,
                 observation_shape: Tuple = (1, 84, 84),
                 observation_dtype: torch.dtype = torch.uint8,
                 history_size: int = 4,
                 gamma: int = 0.9,
                 min_replay_history: int = 20000,
                 update_period: int = 4,
                 target_update_period: int = 8000,
                 epsilon_fn=linearly_decaying_epsilon,
                 epsilon_start: float = 1.0,
                 epsilon_final: float = 0.1,
                 epsilon_decay_period: int = 250000,
                 memory_buffer_size: int = 1000000,
                 minibatch_size: int = 32,
                 device: str = 'cpu',
                 summary_writer=None):
        """
        Initialize the DQN agent

        TODO for future:
            - Add: update horizon (for n-step updates), summary writer,
            - Add: optimizer parameters, network parameters

        :param num_actions: number of actions the agent can take at any state.
        :param observation_shape: tuple of ints describing the observation shape.
        :param observation_dtype: NOTE: type of observaiton
        :param history_size: int, number of observations to use in state stack.
        :param gamma: decay constant
        :param min_replay_history: number of transitions that should be
            experienced before the agent begins training its value function
        :param update_period: int, number of actions between network training
        :param target_update_period: update period of target network (per
            parameter updates)
        :param epsilon_fn: epsilon decay function
        :param epsilon_start: exploration rate at start
        :param epsilon_final: final exploration rate
        :param epsilon_decay_period: length of the epsilon decay schedule
        :param memory_buffer_size: size of the memory buffer for replay
        :param device: 'cuda' or 'cpu', depending on if 'cuda' is available
        :param summary_writer: TODO implement this with TensorBoard in the future
        """

        # Definable attributes
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.history_size = history_size
        self.gamma = gamma
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.target_update_period = target_update_period
        self.epsilon_fn = epsilon_fn
        self.epsilon_start = epsilon_start  # TODO: use this if needed, not used currently
        self.epsilon_final = epsilon_final
        self.epsilon_decay_period = epsilon_decay_period
        self.summary_writer = summary_writer  # TODO: implement this

        self.memory_buffer_size = memory_buffer_size
        self.minibatch_size = minibatch_size
        self.device = device

        # Additional attributes
        memory_buffer_device = self.device  # store buffer in GPU ram

        # Counter attributes
        self.training_steps = 0  # for epsilon decay
        self.total_param_updates = 0

        # Initialize network, memory and optimizer
        self.policy_net = None
        self.target_net = None
        self.memory = None

        self._init_network()
        self._init_memory()

        # https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                       lr=0.00025,
                                       alpha=0.95,
                                       momentum=0.0,  # TODO: 0 or 0.95?
                                       eps=0.00001,
                                       centered=True)

        # History queue: for stacking observations (np matrices in cpu)
        self.history_queue = deque(maxlen=self.history_size)

    def _init_network(self) -> None:
        """Initialize the Q network"""
        self.policy_net = network.nature_dqn_network(self.num_actions,
                                                     self.history_size,
                                                     self.observation_shape[-2:])
        self.target_net = network.nature_dqn_network(self.num_actions,
                                                     self.history_size,
                                                     self.observation_shape[-2:])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target net to evaluation mode

    def _init_memory(self) -> None:
        """Initialize the memory buffer"""
        self.memory = replay_memory.CircularReplayBuffer(
            buffer_cap=self.memory_buffer_size,
            history=self.history_size,
            obs_shape=self.observation_shape,
            device=self.device)

    def begin_episode(self) -> None:
        """
        Prepare the agent at the beginning of a new episode
        """
        # Initialize (zero-padded) history
        for _ in range(self.history_size):
            zero_pad_mat = torch.zeros(self.observation_shape,
                                       dtype=self.observation_dtype,
                                       device='cpu')
            self.history_queue.append(zero_pad_mat)

        # NOTE: first observation is not stored in memory buffer
        # TODO: maybe add training step?

    def step(self, observation: np.ndarray) -> int:
        """
        The agent takes one step

        :param observation:  initial observation from the environment, should
                have shape self.observation_shape
        :return: int denoting action to take at this step
        """
        # Cast observation to tensor and add to history
        observation = torch.tensor(observation, dtype=self.observation_dtype,
                                   device='cpu')
        self.history_queue.append(observation)

        # Select action and return int
        return self._select_action()

    def _select_action(self) -> int:
        """
        Select action according to the epsilon greedy policy
        :return: int denoting action to take
        """

        # Compute epsilon
        epsilon = self.epsilon_fn(self.epsilon_decay_period,
                                  self.training_steps,
                                  self.min_replay_history,
                                  self.epsilon_final)

        if random.random() <= epsilon:
            # random action with probability epsilon
            return random.randrange(self.num_actions)
        else:
            # Construct state
            state = self._history_queue_to_state()
            # greedy action
            action_tensor = self.policy_net(state).max(1)[1].view(1, 1)
            return action_tensor.item()  # TODO ensure this works

    def _history_queue_to_state(self) -> torch.tensor:
        """
        Convert the current history queue into a torch tensor state, where the
        state is sent to device (same as the neural nets)

        :return: 4-dimensional torch.tensor object of state, consisting of
                dims: [1, history size, height, width]
        """
        state = torch.cat(list(self.history_queue), dim=0).unsqueeze(0)
        state = state.type(torch.float32).to(self.device)
        return state

    def store_transition(self, action: int, observation: np.ndarray,
                         reward: float, is_terminal: bool) -> None:
        """
        Stores the recently experienced transition to the memory buffer

        :param action: action taken
        :param observation: np array of received observation after action
        :param reward: float of reward received as result of action
        :param is_terminal: is episode finished?
        """

        # Cast all to tensor
        action = torch.tensor([action], dtype=torch.int32, device=device)
        observation = torch.tensor(observation, dtype=self.observation_dtype,
                                   device=device)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        is_terminal = torch.tensor([is_terminal], dtype=torch.bool,
                                   device=device)

        # TODO: reward clipping not done here, do in environment
        # Push to memory
        self.memory.push(observation, action, reward, is_terminal)

    def optimize_model(self) -> float:
        """
        Optimizes the model for
        :return:
        """
        # If not enough memory
        if len(self.memory) < self.minibatch_size:
            return -1.0

        # Sample memory and unpack
        mem_batch = self.memory.sample(self.minibatch_size)
        state_batch, action_batch, reward_batch, next_state_batch = mem_batch

        state_batch = state_batch.type(torch.float32).to(self.device)
        action_batch = action_batch.type(torch.long).to(self.device)
        reward_batch = reward_batch.type(torch.float32).to(self.device)
        next_state_batch = next_state_batch.type(torch.float).to(self.device)

        # Compute current and expected values
        state_action_values = self.policy_net(state_batch) \
            .gather(1, action_batch)  # esti vals for acts taken, size: (batch-size, 1)
        next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute TD loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # gradient-clipping
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        self.total_param_updates += 1
        # Update policy net
        if self.total_param_updates % self.target_update_period == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Return loss
        if loss is not None:
            return loss
        else:
            return -1.0


if __name__ == "__main__":
    # for testing run this directly
    print('testing')
    agent = DQNAgent(num_actions=8)
    print(agent)
    print(agent.policy_net)
    print(agent.target_net)
    print(agent.memory.capacity)
