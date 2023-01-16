import numpy as np


############################
# Define helper functions  #
############################
# Helper functions are for consistency between different MDP settings
# The complex operations for the model assisted weight method interface with
#   various object classes
# Helper functions define the MDP specific behavior while maintaining the same
#   interface for the object classes
# The helper functions should not need to be called by the user


def draw(probs):
    """
    This function draws from 0,...,len(probs) according to the supplied probabilities

    Parameters
    ----------
    probs : 1d numpy array
        Vector of probabilities .

    Returns
    -------
    float
        Random number between 0 and len(probs).

    """
    # normalize probabilities to sum to 1
    probs = probs / np.sum(probs)
    # get cumulative probabilities
    cu_probs = np.cumsum(probs)
    # dram random uniform variate
    rand_unif = np.random.uniform(0, 1, 1)
    # loop over cumulative probabilities
    # return first that is > rand_unif
    for i in range(probs.size):
        if rand_unif < cu_probs[i]:
            # return as float
            return i * 1.0
    # extra return statement in case of numerical precision issues
    return probs.size - 1


def _get_next_state(state, action, params):
    """
    Get the next state of the MDP based on the current state and action

    Parameters
    ----------
    state : 1d numpy array
        Current state of the MDP.
    action : float
        The action to be applied to the MDP. can be either 0.0 or 1.0.
    params : 3d numpy array
        Parameters governing transition dynamics.

    Returns
    -------
    1d numpy array
        The next state of the MDP.

    """
    # extract appropriate probabilities from params based on action
    if action == 0:
        trans_prob = params[0, int(state[0]), :]
    else:
        trans_prob = params[1, int(state[0]), :]
    # draw next state according to probabilities
    next_state = draw(trans_prob)
    # return as numpy array
    return np.array([next_state])


def _get_initial_state(dim):
    """
    Get the initial state of the MDP

    Parameters
    ----------
    dim : int
        The number of unique states in state space.

    Returns
    -------
    1d numpy array
        The initial state of the MDP.

    """
    # draw randomly with equal probability
    return np.array([draw(np.array([1] * dim) / dim)])


# noinspection PyUnusedLocal
def _get_reward(state, action, next_state, reward_vec):
    """
    Get the real value reward signal for the MDP

    Parameters
    ----------
    state : 1d numpy array
        Current state of the MDP.
    action : float
        The action to be applied to the MDP. can be either 0.0 or 1.0.
    next_state : 1d numpy array
        Next state of the MDP.
    reward_vec : 1d numpy array
        vector containing reward values.

    Returns
    -------
    reward_val : float
        Reward for given state, action, next state triple.

    """
    reward_val = reward_vec[int(next_state[0])] - 0.5 * action
    return reward_val


def _fit_transition_model(data, dim):
    """
    Estimate the parameters for the transition dynamics model based on data

    Parameters
    ----------
    data : 2d numpy array
        Matrix of data extracted from DataObject class.
    dim : int
        The number of unique states in state space.

    Returns
    -------
    params : 3d numpy array
        Parameters governing transition dynamics.

    """
    # initialize parameter array
    #  dimension 1 is action
    #  dimension 2 is current state
    #  dimension 3 is next state
    params = np.zeros((2, dim, dim))
    # loop over each data point to count transitions for each state, action, next state combination
    for i in range(data.shape[0]):
        params[int(data[i, 3]), int(data[i, 2]), int(data[i, 5])] += 1
    # normalize across rows to get probabilities
    for i in range(dim):
        params[0, i, :] = params[0, i, :] / np.sum(params[0, i, :])
        params[1, i, :] = params[1, i, :] / np.sum(params[1, i, :])
    return params


####################################
# Define the TransitionModel class #
####################################

class TransitionModel:
    """
    A class to define the transition dynamics model for an MDP
    
    Attributes:
       reward_vec : 1d numpy array
           Vector containing reward values
       dim : int
           Number of unique states in state space
       params : 3d numpy array 
           Parameters governing transition dynamics
           
    Methods:   
       __init__(reward_vec, dim)
       setParameters(params)
       getParameters()
       getInitialState()
       fitTransitionModel(data)
       getNextStates(state, action)
       getReward(state, action, nextState)
    
    Notes:
        Method interfaces are standardized.
        Any changes to behaviour need to be handled by adding
        more to __init__ function or with the helper functions.
    """

    def __init__(self, reward_vec, dim=3):
        """
        Initialize object
            
        Parameters
        ----------
        reward_vec : 1d numpy array
            Vector containing reward values for MDP.
        dim : int, optional
             The number of unique states in state space. The default is 3.

        Returns
        -------
        None.
        
        Notes
        -----
        Creates 'params' placeholder for parameters

        """
        self.dim = dim
        self.reward_vec = reward_vec
        self.params = np.zeros((2, dim, dim))

    def set_parameters(self, params):
        """
        Set the value of the parameters for the transition dynamics model

        Parameters
        ----------
        params : 3d numpy array
            Parameters governing transition dynamics.

        Returns
        -------
        None.

        """
        assert params.shape == (2, self.dim, self.dim)
        params = params.copy()
        for i in range(self.dim):
            params[0, i, :] = params[0, i, :] / np.sum(params[0, i, :])
            params[1, i, :] = params[1, i, :] / np.sum(params[1, i, :])
        self.params = params

    def get_parameters(self):
        """
        Returns the parameters for the transition dynamics model

        Returns
        -------
        3d numpy array
            Parameters governing transition dynamics.

        """
        return self.params

    def get_initial_state(self):
        """
        Randomly set the initial state of the MDP

        Returns
        -------
        1d numpy array
            Initial state of the MDP.

        """
        return _get_initial_state(self.dim)

    def fit_transition_model(self, data):
        """
        Estimate the parameters for the transition model from data.

        Parameters
        ----------
        data : 2d numpy array
            An array extracted from the DataObject class.

        Returns
        -------
        None.

        """
        self.params = _fit_transition_model(data, self.dim)

    def get_next_state(self, state, action):
        """
        Update the state of the MPD based on the current state and selected action

        Parameters
        ----------
        state : 1d numpy array
            Current state of the MDP.
        action : float
            The selected action. Can be either 0.0 or 1.0.

        Returns
        -------
        1d numpy array
            Next state of the MDP.

        """
        return _get_next_state(state, action, self.get_parameters())

    def get_reward(self, state, action, next_state):
        """
        Get the real value reward signal for the MDP

        Parameters
        ----------
        state : 1d numpy array
            Current state of the MDP.
        action : float
            The action to be applied to the MDP. can be either 0.0 or 1.0.
        next_state : 1d numpy array
            Next state of the MDP.
    
        Returns
        -------
        float
            Reward for provided state, action, nextState triple.

        """
        return _get_reward(state, action, next_state, self.reward_vec)
