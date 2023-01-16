import numpy as np


def _get_feature(state):
    """
    Get the feature space for a single state

    Parameters
    ----------
    state : 1d numpy array
        The state of the MDP.

    Returns
    -------
    features : 1d numpy array
        The feature space of the supplied state.

    """
    features = np.ones(2)
    features[1] = state[0] - 1
    return features


def _get_features(states):
    """
    Get the feature space for multiple states

    Parameters
    ----------
    states : 2d numpy array
        Several states of the MDP.

    Returns
    -------
    features : 2d numpy array
        The feature space expansion of the supplied states.

    """
    n = states.shape[0]
    features = np.zeros((n, 2))
    for i in range(n):
        features[i, :] = _get_feature(states[i, :])
    return features


class FeatureModel:
    def __init__(self):
        """
        Initialize feature space object

        Parameters
        ----------
        None

        Returns
        -------
        None.

        """

    @staticmethod
    def get_features_current(data):
        """
        Get the features for the current states in the data

        Parameters
        ----------
        data : DataObject
            An object of the class type DataObject.

        Returns
        -------
        2d numpy array
            The feature space expansion of the current states.

        """
        return _get_features(data.get_states())

    @staticmethod
    def get_features_next(data):
        """
        Get the features for the next states in the data

        Parameters
        ----------
        data : DataObject
            An object of the class type DataObject.

        Returns
        -------
        2d numpy array
            The feature space expansion of the next states.

        """
        return _get_features(data.get_next_states())

    @staticmethod
    def get_features(states):
        """
        Get the feature space for multiple states

        Parameters
        ----------
        states : 2d numpy array
            Several states of the MDP.

        Returns
        -------
        features : 2d numpy array
            The feature space expansion of the supplied states.

        """
        return _get_features(states)

    @staticmethod
    def get_feature(state):
        """
        Get the feature space for a single state

        Parameters
        ----------
        state : 1d numpy array
            The state of the MDP.

        Returns
        -------
        features : 1d numpy array
            The feature space of the supplied state.

        """
        return _get_feature(state)
