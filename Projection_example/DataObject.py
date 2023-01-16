import numpy as np

############################
# Define helper functions  #
############################
# Helper functions are for consistency between different MDP settings
# The complex operations for the model assisted weight method interface with
#   various object classes
# Helper functions define the MDP specific behavior while maintain the same
#   interface for the object classes
# The helper functions should not need to be called by the user


def _generate(n_sub, n_obs, state_dim, policy_object, transition_model, random_seed):
    """
    Generate data using transitionModel and policyObject

    Parameters
    ----------
    n_sub : int
        Number of subjects.
    n_obs : int
        Number of observations per subject.
    state_dim : int
        Number of variables needed to define the state.
    policy_object : PolicyObject class
        An object containing the action selection policy.
    transition_model : TransitionModel class
        An object containing the transition dynamics model.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : 2d numpy array
        An array containing the data. The size of the array should be (n_sub*n_obs) by (2*stateDim + 5)

    Notes
    -----
    The columns of the data are:
        subject #, observation #, <state>, action. probability, <next_state>, reward
        <state> and <next_state> will each occupy stateDim columns

    """
    np.random.seed(random_seed)
    # create data array
    data = np.zeros(((n_sub * n_obs), (state_dim * 2 + 5)))
    # initialize counter to determine current row to assign to
    counter = 0
    next_state = np.zeros(state_dim)
    for i in range(n_sub):
        for j in range(n_obs):
            # if first observation on a subject get initial state
            if j == 0:
                state = transition_model.get_initial_state()
            # otherwise copy state from next_state on last iteration
            else:
                state = next_state.copy()
            # select action
            action, prob = policy_object.get_action(state)
            # update state
            next_state = transition_model.get_next_state(state, action)
            # calculate reward
            reward = transition_model.get_reward(state, action, next_state)
            # convert state and next_state to list
            state_list = [state[0]]
            next_list = [next_state[0]]
            for k in range(1, state_dim):
                # add in each dimension of state
                state_list += [state[k]]
                next_list += [next_state[k]]
            # Assign data to row
            data[counter, :] = [i, j]+state_list+[action, prob]+next_list+[reward]
            counter += 1
    return data


class DataObject:
    """
    A class for holding, querying, and generating data

    Attributes:
        stateDim : int
            Number of variables needed to define the state.
        randomSeed : int, optional
            Random seed for reproducibility. The default is 1.
    Methods:
        generate(n_sub, n_obs, policyObject, transitionModel, random_seed)
        setData(data)
        getData()
        getStates()
        getNextStates()
        getActions()
        getProbs()
        getRewards()

    Notes:
        Method interfaces are standardized.
        Any changes to behaviour need to be handled by adding
        more to __init__ function or with the helper functions.
    """
    
    def __init__(self, state_dim, random_seed=1):
        """
        Initialize DataObject

        Parameters
        ----------
        state_dim : int
            Number of variables needed to define the state.
        random_seed : int, optional
            Random seed for reproducibility. The default is 1.

        Returns
        -------
        None.

        """
        self.n_obs = None
        self.n_sub = None
        self.data = None
        self.stateDim = state_dim
        self.randomSeed = random_seed
    
    def generate(self, n_sub, n_obs, policy_object, transition_model, random_seed=None):
        """
        Generate data using transitionModel and policyObject

        n_sub : int
            Number of subjects.
        n_obs : int
            Number of observations per subject.
        policyObject : PolicyObject class
            An object containing the action selection policy.
        transitionModel : TransitionModel class
            An object containing the transition dynamics model.
        random_seed : int, optional
            Random seed for reproducibility. The default is 1.

        Returns
        -------
        None.

        """
        if random_seed is None:
            random_seed = self.randomSeed
        self.n_sub = n_sub
        self.n_obs = n_obs
        self.data = _generate(self.n_sub, self.n_obs, self.stateDim, policy_object, transition_model, random_seed)
      
    def set_data(self, data, n_sub, n_obs):
        """
        Set the data. For splitting data for cross validation.

        Parameters
        ----------
        data : 2d numpy array
            An array containing the data. The size of the array should be (n_sub*n_obs) by (2*stateDim + 5)
        n_sub : int
            Number of subjects.
        n_obs : int
            Number of observations per subject.

        Returns
        -------
        None.

        """
        self.n_sub = n_sub
        self.n_obs = n_obs
        self.data = data

    def get_data(self):
        """
        Return the data array

        Returns
        -------
        2d numpy array
            The array containing the data.

        """
        return self.data

    def get_states(self):
        """
        Return the states from the data

        Returns
        -------
        2d numpy array
            The array of states.

        """
        return self.data[:, 2:2+self.stateDim]

    def get_next_states(self):
        """
        Return the next states from the data

        Returns
        -------
        2d numpy array
            The array of nextStates.

        """
        return self.data[:, 4+self.stateDim:4+2*self.stateDim]

    def get_actions(self):
        """
        Return the actions from the data

        Returns
        -------
        1d numpy array
            An array of the actions.

        """
        return self.data[:, 2+self.stateDim]

    def get_probs(self):
        """
        Return the probabilities of the actions from the data

        Returns
        -------
        1d numpy array
            An array of the probabilities.

        """
        return self.data[:, 3+self.stateDim]
    
    def get_rewards(self):
        """
        Return the rewards from the data

        Returns
        -------
        1d numpy array
            An array of the rewards.

        """
        return self.data[:, 4+2*self.stateDim]
