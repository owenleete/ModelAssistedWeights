import numpy as np
from numba import njit


# _get_policy_feature_space(state) - Helper function for PolicyObject class
# Input:
#   state - A numpy array describing the current state of the environment
# Output:
#   features - A numpy array with components needed to calculate the action probability
@njit
def _get_policy_feature_space(state):
    return np.array([1.0, state[0], state[1], state[0] * state[1]])


# _get_action(state,params) - Helper function for PolicyObject class
# Input:
#   state - A numpy array describing the current state of the environment
#   params - A numpy array containing all information needed to determine action probabilities from a given state
# Outputs:
#   action - The action to recommend
#   prob - The probability of selecting the recommended action
@njit
def _get_action(state, params):
    lp = np.sum(_get_policy_feature_space(state) * params)
    # Truncate to avoid Inf
    if lp > 500:
        lp = 500
    prob = np.exp(lp) / (1 + np.exp(lp))
    action = np.random.binomial(1, prob)
    if action == 0:
        return action, 1 - prob
    else:
        return action, prob


# _get_prob(state, action, params) - Helper function for PolicyObject class
# Input:
#   state - A numpy array describing the current state of the environment
#   action - The action that was recommended
#   params - A numpy array containing all information needed to determine action probabilities from a given state
# Output:
#   prob - The probability of selecting the given action
@njit
def _get_prob(state, action, params):
    lp = np.sum(_get_policy_feature_space(state) * params)
    # Truncate to avoid Inf
    if lp > 500:
        lp = 500
    if action == 1:
        return np.exp(lp) / (1 + np.exp(lp))
    return 1 - np.exp(lp) / (1 + np.exp(lp))


# _get_probs(states, actions, params) - Helper function for the PolicyObject class
# Input:
#   states - A 2d numpy array containing several state vectors
#   actions - A numpy array containing the actions that were taken
#   params - A numpy array containing all information needed to determine action probabilities from a given state
# Output:
#   probs - The probabilities that the supplied actions would have been recommended by the current policy conditional on
#            the supplied states
@njit
def _get_probs(states, actions, params):
    n = states.shape[0]
    probs = np.zeros(n)
    for i in range(n):
        probs[i] = _get_prob(states[i, :], actions[i], params)
    return probs


# A class defining the behavior of the policy for a Markov decision process (MDP)
# Attributes:
#   parameters - A numpy array containing all information needed to determine action probabilities from a given state
# Methods:
#   get_action(state)
#   getProbabilities(states, actions)
class PolicyObject:
    def __init__(self, params):
        self.params = params

    # get_action(state) - Returns a recommended action and the probability of selection
    # Inputs:
    #   state - A numpy array describing the current state of the environment
    # Output:
    #   action - The recommended action 
    #   prob - The probability of selecting the recommended action
    def get_action(self, state):
        return _get_action(state, self.params)

    # get_probs(states,actions) - Returns the probabilities of selecting the actions that were taken
    # Input:
    #   states - A 2d numpy array containing several state vectors
    #   actions - A numpy array containing the actions that were taken
    # Output:
    #   probs - The probabilities that the supplied actions would have been recommended by the current policy
    #            conditional on the supplied states
    def get_probs(self, states, actions):
        return _get_probs(states, actions, self.params)
