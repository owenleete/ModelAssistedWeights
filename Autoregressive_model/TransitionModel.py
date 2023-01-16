import numpy as np
from numba import njit


# _get_next_state(state, action, theta1, theta2, sd1, sd2) - Helper function for TransitionModel class.
#           Returns subsequent state
# Input:
#   state - A numpy array describing the current state of the environment
#   action - The action that was applied
#   theta1 - numpy array containing parameters to determine mean of the first component of nextState
#   theta2 - numpy array containing parameters to determine mean of the second component of nextState
#   sd1 - float containing parameter that determines standard deviation of the first component of nextState
#   sd2 - float containing parameter that determines standard deviation of the second component of nextState
# Output:
#   nextState - A numpy array containing the next state
@njit
def _get_next_state(state, action, theta1, theta2, sd1, sd2):
    # Create covariates
    x = np.zeros((1, 7))
    x[0, 0] = 1
    x[0, 1] = state[0]
    x[0, 2] = state[1]
    x[0, 3] = state[0] * state[1]
    x[0, 4] = action
    x[0, 5] = state[0] * action
    x[0, 6] = state[1] * action

    # Calculate next state
    state1 = np.sum(x * theta1) + np.random.normal(0, sd1)
    state2 = np.sum(x * theta2) + np.random.normal(0, sd2)

    # Truncate to avoid infinite values
    if state1 > 999.0:
        state1 = 999.0
    elif state1 < -999.0:
        state1 = -999.0

    if state2 > 999.0:
        state2 = 999.0
    elif state2 < -999.0:
        state2 = -999.0

    return np.array([state1, state2])


# _get_initial_state(theta1, theta2, sd1, sd2) - Helper function for TransitionModel class. Returns a starting state
#           for a new subject
# Input:
#   theta1 - A numpy array containing parameters to determine mean of the first component of nextState
#   theta2 - A numpy array containing parameters to determine mean of the second component of nextState
#   sd1 - A float containing parameter that determines standard deviation of the first component of nextState
#   sd2 - A float containing parameter that determines standard deviation of the second component of nextState
# Output:
#   state - A numpy array describing the initial state of the environment
@njit
def _get_initial_state(theta1, theta2, sd1, sd2):
    state = np.random.normal(0, 1, 2)
    for k in range(50):
        action = np.random.binomial(1, 0.5)
        state = _get_next_state(state, action, theta1, theta2, sd1, sd2)
        return state


# _get_reward(state, action, nextState) - Helper function for TransitionModel class. Calculates the reward for a
#           state-action-next state triple
# Input:
#   state - environment state at time t
#   action - action taken
#   nextState - environment state at time t+1
# Output:
#   reward - relative preference of state-action-next state triple
# The unused 'state' is left in to be compatible with the general format for reward functions
# noinspection PyUnusedLocal
@njit
def _get_reward(state, action, next_state):
    return 2 * next_state[0] + next_state[1] - (1 / 4) * (2 * action - 1)


# _fit_full_model(data) - Fit the correct transition model
# Input:
#   data - A 2d numpy array containing the data
# Output:
#   theta1 - A numpy array containing parameters to determine mean of the first component of nextState
#   theta2 - A numpy array containing parameters to determine mean of the second component of nextState
#   sd1 - A float containing parameter that determines standard deviation of the first component of nextState
#   sd2 - A float containing parameter that determines standard deviation of the second component of nextState
def _fit_full_model(data):
    targets = data.get_next_states()
    y1 = targets[:, 0]
    y2 = targets[:, 1]
    variables = data.get_states()
    actions = data.get_actions()
    x = np.zeros((variables.shape[0], 7))
    x[:, 0] = 1
    x[:, 1] = variables[:, 0]
    x[:, 2] = variables[:, 1]
    x[:, 3] = variables[:, 0] * variables[:, 1]
    x[:, 4] = actions
    x[:, 5] = variables[:, 0] * actions
    x[:, 6] = variables[:, 1] * actions
    theta1 = np.matmul(np.linalg.inv(np.inner(x.T, x.T)), np.inner(x.T, y1))
    theta2 = np.matmul(np.linalg.inv(np.inner(x.T, x.T)), np.inner(x.T, y2))
    y1_hat = np.matmul(x, theta1)
    y2_hat = np.matmul(x, theta2)
    sd1 = np.sqrt(np.mean((y1_hat - y1) ** 2))
    sd2 = np.sqrt(np.mean((y2_hat - y2) ** 2))
    return theta1, theta2, sd1, sd2


# A class for holding, querying, and generating data
# Attributes:
#   theta1 - A numpy array containing parameters to determine mean of the first component of nextState
#   theta2 - A numpy array containing parameters to determine mean of the second component of nextState
#   sd1 - A float containing parameter that determines standard deviation of the first component of nextState
#   sd2 - A float containing parameter that determines standard deviation of the second component of nextState
#   modelIdentifier - An int indicating which model to fit (0 - Correct model,
#                                                           1 - model with squared terms,
#                                                               but no interaction between state components,
#                                                           2 - Model with only main effects)
# Methods:   
#   __init__
#   set_parameters(theta1, theta2, sd1, sd2)
#   get_parameters()
#   get_initial_state()
#   fit_transition_model()
#   get_next_states()
#   get_reward()
class TransitionModel:
    def __init__(self):
        self.theta1 = None
        self.theta2 = None
        self.sd1 = None
        self.sd2 = None

    # set_parameters() - Sets the transition model parameters to the provided values
    # Input:
    #   theta1 - A numpy array containing parameters to determine mean of the first component of nextState
    #   theta2 - A numpy array containing parameters to determine mean of the second component of nextState
    #   sd1 - A float containing parameter that determines standard deviation of the first component of nextState
    #   sd2 - A float containing parameter that determines standard deviation of the second component of nextState
    # Output:
    #   None
    def set_parameters(self, theta1, theta2, sd1, sd2):
        self.theta1 = theta1
        self.theta2 = theta2
        self.sd1 = sd1
        self.sd2 = sd2

    # get_parameters() - Returns the transition model parameters
    # Input:
    #   None
    # Output:
    #   theta1 - A numpy array containing parameters to determine mean of the first component of nextState
    #   theta2 - A numpy array containing parameters to determine mean of the second component of nextState
    #   sd1 - A float containing parameter that determines standard deviation of the first component of nextState
    #   sd2 - A float containing parameter that determines standard deviation of the second component of nextState  
    def get_parameters(self):
        return self.theta1, self.theta2, self.sd1, self.sd2

    # _get_initial_state(theta1, theta2, sd1, sd2) - Helper function for TransitionModel class.
    #       Returns a starting state for a new subject
    # Input:
    #   None
    # Output:
    #   state - A numpy array describing the initial state of the environment
    def get_initial_state(self):
        return _get_initial_state(*self.get_parameters())

    # _fit_full_model(data) - Fits the transition model
    # Input:
    #   data - A 2d numpy array containing the data
    # Output:
    #   None
    def fit_transition_model(self, data):
        self.theta1, self.theta2, self.sd1, self.sd2 = _fit_full_model(data)

    # get_next_state(state, action) - Returns subsequent state
    # Input:
    #   state - A numpy array describing the current state of the environment
    #   action - The action that was applied
    # Output:
    #   nextState - A numpy array containing the next state
    def get_next_state(self, state, action):
        return _get_next_state(state, action, *self.get_parameters())

    # get_reward(state, action, nextState) - Calculates the reward for a state-action-next state triple
    # Input:
    #   state - environment state at time t
    #   action - action taken
    #   nextState - environment state at time t+1
    # Output:
    #   reward - relative preference of state-action-next state triple
    @staticmethod
    def get_reward(state, action, next_state):
        return _get_reward(state, action, next_state)
