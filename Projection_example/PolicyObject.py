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


def _get_action(state, params):
    """
    Randomly draw the action based on the current state

    Parameters
    ----------
    state : 1d numpy array
        Current state of the MDP.
    params : 1d numpy array
        Parameters governing action selection policy.

    Returns
    -------
    action : int
        The action that was selected.
    prob : float
        The probability of selecting the action.
        
    Notes
    -----
    If no parameters are specified use the equiprobable policy
    """
    if params is not None:
        lp = params[int(state[0])]
        # truncate lp to avoid Inf
        lp = min((500., lp))
        # get probability of selecting action = 1
        prob1 = np.exp(lp)/(1+np.exp(lp))
        rand_unif = np.random.uniform(0, 1, 1)
        if rand_unif < prob1:
            # chose action 1
            action = 1 
            prob = prob1
        else:
            # chose action 0
            action = 0
            prob = 1-prob1
        return action, prob
    else:
        rand_unif = np.random.uniform(0, 1, 1)
        if rand_unif < 0.5:
            action = 0 
            prob = 0.5
        else:
            action = 1
            prob = 0.5
        return action, prob


def _get_prob(state, action, params):
    """
    Calculate the probability of selecting the given action

    Parameters
    ----------
    state : 1d numpy array
        Current state of the MDP.
    action : int
        The action that was selected.
    params : 1d numpy array
        Parameters governing action selection policy.

    Returns
    -------
    prob : float
        The probability of selecting the action.
        
    Notes
    -----
    If no parameters are specified use the equiprobable policy
    """
    if params is not None:
        lp = params[int(state[0])]
        # truncate lp to avoid Inf
        lp = min((500., lp))
        # calculate probability
        prob1 = np.exp(lp)/(1+np.exp(lp))
        if action == 1:
            return prob1
        else:
            return 1-prob1
    else:
        return 0.5


def _get_probs(states, actions, params):
    """
    Calculate the probability for multiple states and actions

    Parameters
    ----------
    states : 2d numpy array
        States of the MDP.
    actions : 1d numpy array
        Action selected for each state.
    params : 1d numpy array
        Parameters governing action selection policy.

    Returns
    -------
    probs : 1d numpy array
        Probabilities for each state-action pair.

    """
    n = states.shape[0]
    probs = np.zeros(n)
    for i in range(n):
        probs[i] = _get_prob(states[i, :], actions[i], params)
    return probs


class PolicyObject:
    """
    A class to define the action selection policy for an MDP
    
    Attributes:
       params : 1d numpy array
           Parameters governing action selection policy
           
    Methods:   
       __init__(params)
       getAction(state)
       getProb(state, action)
       getProbs(states, actions)
       
    Notes:
        Method interfaces are standardized.
        Any changes to behaviour need to be handled by adding
        more to __init__ function or with the helper functions.
    """
    def __init__(self, params):
        """
        Initialize object

        Parameters
        ----------
        params : 1d numpy array
        Parameters governing action selection policy.

        Returns
        -------
        None.

        """
        self.params = params
    
    def get_action(self, state):
        """
        Randomly draw the action based on the current state
        
        Parameters
        ----------
        state : 1d numpy array
            Current state of the MDP.
    
        Returns
        -------
        action : int
            The action that was selected.
        prob : float
            The probability of selecting the action.
            
        Notes
        -----
        If no parameters are specified use the equiprobable policy

        """
        return _get_action(state, self.params)
    
    def get_prob(self, state, action):
        """
        Calculate the probability of selecting the given action
    
        Parameters
        ----------
        state : 1d numpy array
            Current state of the MDP.
        action : int
            The action that was selected.
    
        Returns
        -------
        prob : float
            The probability of selecting the action.
            
        Notes
        -----
        If no parameters are specified use the equiprobable policy

        """
        return _get_prob(state, action, self.params)
    
    def get_probs(self, states, actions):
        """
        Calculate the probability for multiple states and actions
    
        Parameters
        ----------
        states : 2d numpy array
            States of the MDP.
        actions : 1d numpy array
            Action selected for each state.
    
        Returns
        -------
        probs : 1d numpy array
            Probabilities for each state-action pair.

        """
        return _get_probs(states, actions, self.params)
    