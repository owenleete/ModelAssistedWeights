import numpy as np
from numba import njit

# Import helper functions from PolicyObject.py and TransitionModel.py to allow speedup with @njit
from PolicyObject import _get_action
from TransitionModel import _get_initial_state, _get_next_state, _get_reward


# _generate(nSub, nObs, stateDim, policyParams, theta1, theta2, sd1, sd2, randomSeed) - Helper function DataObject class
# Input:
#   nSub - Number of subjects to simulate
#   nObj - Number of observations per subject
#   stateDim - Dimension of the state vector
#   policyParams - numpy array containing all information needed to determine action probabilities
#   theta1 - numpy array containing part of the transition model parameters
#   theta2 - numpy array containing part of the transition model parameters
#   sd1 - float containing part of the transition model parameters
#   sd2 - float containing part of the transition model parameters
#   randomSeed - value used to seed the random number generator
# Output:
#   data - A 2d numpy array of dimensions (nSub*nObs) by (2*stateDim+5)
@njit
def _generate(n_sub, n_obs, state_dim, policy_params, theta1, theta2, sd1, sd2, random_seed):
    np.random.seed(random_seed)
    data = np.zeros(((n_sub * n_obs), (state_dim * 2 + 5)))
    counter = 0
    next_state = np.zeros(state_dim)
    for i in range(n_sub):
        for j in range(n_obs):
            if j == 0:
                state = _get_initial_state(theta1, theta2, sd1, sd2)
            else:
                state = next_state.copy()
            action, prob = _get_action(state, policy_params)
            next_state = _get_next_state(state, action, theta1, theta2, sd1, sd2)
            reward = _get_reward(state, action, next_state)
            state_list = [state[0]]
            next_list = [next_state[0]]
            for k in range(1, state_dim):
                state_list += [state[k]]
                next_list += [next_state[k]]
            data[counter, :] = [i, j] + state_list + [action, prob] + next_list + [reward]
            counter += 1
    return data


# A class for holding, querying, and generating data
# Attributes:
#   stateDim - The number of elements in the state vector
#   randomSeed - The seed for the random number generator
#   nSub - The number of subjects
#   nObs - The number of observations per subjects
#   data - A 2d numpy array containing the data
# Methods:   
#   __init__
#   generate(nSub, nObs, stateDim, policyParams, theta1, theta2, sd1, sd2 randomSeed)
#   set_data(data, nSub, nObs)
#   get_data()
#   get_states()
#   get_next_states()
#   get_actions()
#   get_probs()
#   get_rewards()
class DataObject:
    def __init__(self, state_dim, random_seed=1):
        self.data = None
        self.n_sub = None
        self.n_obs = None
        self.state_dim = state_dim
        self.randomSeed = random_seed

    # generate(nSub, nObs, policyObject, transitionModel, randomSeed) - generates a data set based on the supplied
    #                                                                    policy and transition model
    # Input:
    #   nSub - The number of subjects
    #   nObs - The number of observations per subjects
    #   policyObject - Object used to determine action probabilities
    #   transitionModel - Object used to determine state transitions
    #   randomSeed - Seed for random number generator. If not supplied it defaults to the DataObject's internal random
    #                 seed
    # Output:
    #   None - sets internal attribute self.data        
    def generate(self, n_sub, n_obs, policy_object, transition_model, random_seed=None):
        if random_seed is None:
            random_seed = self.randomSeed
        self.n_sub = n_sub
        self.n_obs = n_obs
        self.data = _generate(self.n_sub, self.n_obs, self.state_dim, policy_object.params,
                              *transition_model.get_parameters(), random_seed)

    # set_data(data, nSub, nObs) - Allows instantiation with external data (useful for cross validation)
    # Input:
    #   data - A 2d numpy array containing the data
    #       Note: Data should contain columns on the order:
    #           Subject index
    #           Observation index
    #           Current State (may take multiple columns)
    #           Action
    #           Probability of action
    #           Next State (may take multiple columns)
    #           Reward
    #   nSub - The number of subjects
    #   nObs - The number of observations per subjects 
    # Output:
    #   None - sets internal attribute self.data   
    def set_data(self, data, n_sub, n_obs):
        self.n_sub = n_sub
        self.n_obs = n_obs
        self.data = data

    # get_data() - Returns full data array
    # Input:
    #   None
    # Output:
    #   data - A 2d numpy array containing the data
    def get_data(self):
        return self.data

    # get_states() - Returns array of current states
    # Input:
    #   None
    # Output:
    #   data - A 2d numpy array containing the current states
    def get_states(self):
        return self.data[:, 2:2 + self.state_dim]

    # get_next_states() - Returns array of subsequent states
    # Input:
    #   None
    # Output:
    #   data - A 2d numpy array containing the next states    
    def get_next_states(self):
        return self.data[:, 4 + self.state_dim:4 + 2 * self.state_dim]

    # get_actions() - Returns array of actions/treatments
    # Input:
    #   None
    # Output:
    #   data - A 1d numpy array containing the actions
    def get_actions(self):
        return self.data[:, 2 + self.state_dim]

    # get_probs() - Returns array of treatment probabilities under generating policy 
    # Input:
    #   None
    # Output:
    #   data - A 1d numpy array containing the action probabilities
    def get_probs(self):
        return self.data[:, 3 + self.state_dim]

    # get_rewards() - Returns array of rewards/utilities
    # Input:
    #   None
    # Output:
    #   data - A 1d numpy array containing the rewards
    def get_rewards(self):
        return self.data[:, 4 + 2 * self.state_dim]
