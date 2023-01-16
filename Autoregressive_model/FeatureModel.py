import numpy as np
from numba import njit


# _get_feature_space_quadratic(state) - Returns the quadratic feature space
# Input:
#   state - A numpy array describing the current state of the environment
# Output:
#   feature vector - A numpy array
@njit
def _get_feature_space_quadratic(state):
    return np.array([1.0, state[0], state[1], state[0] * state[1], state[0] ** 2, state[1] ** 2])


# _get_feature_space_linear(state) - Returns the linear feature space
# Input:
#   state - A numpy array describing the current state of the environment
# Output:
#   feature vector - A numpy array
@njit
def _get_feature_space_linear(state):
    return np.array([1.0, state[0], state[1]])


# _get_feature_space_gaussian(states) - Returns the Gaussian feature space
# Input:
#   states - A 2d numpy array containing state vectors
# Output:
#   feature vector - A numpy array
@njit
def _get_feature_space_gaussian(states, s1max, s1min, s2max, s2min):
    new_states = np.zeros_like(states)
    new_states[:, 0] = (states[:, 0] - s1min) / (s1max - s1min)
    new_states[:, 1] = (states[:, 1] - s2min) / (s2max - s2min)
    features = np.zeros((states.shape[0], 11))
    features[:, 0] = 1
    features[:, 1] = np.exp(-(new_states[:, 0] - 0.00) ** 2 / (2 * 0.25 ** 2))
    features[:, 2] = np.exp(-(new_states[:, 0] - 0.25) ** 2 / (2 * 0.25 ** 2))
    features[:, 3] = np.exp(-(new_states[:, 0] - 0.50) ** 2 / (2 * 0.25 ** 2))
    features[:, 4] = np.exp(-(new_states[:, 0] - 0.75) ** 2 / (2 * 0.25 ** 2))
    features[:, 5] = np.exp(-(new_states[:, 0] - 1.00) ** 2 / (2 * 0.25 ** 2))
    features[:, 6] = np.exp(-(new_states[:, 1] - 0.00) ** 2 / (2 * 0.25 ** 2))
    features[:, 7] = np.exp(-(new_states[:, 1] - 0.25) ** 2 / (2 * 0.25 ** 2))
    features[:, 8] = np.exp(-(new_states[:, 1] - 0.50) ** 2 / (2 * 0.25 ** 2))
    features[:, 9] = np.exp(-(new_states[:, 1] - 0.75) ** 2 / (2 * 0.25 ** 2))
    features[:, 10] = np.exp(-(new_states[:, 1] - 1.00) ** 2 / (2 * 0.25 ** 2))
    return features


# _get_features(states, featureSpace) - Returns the feature space for the model free component
# Input:
#   states - A 2d numpy array containing state vectors
#   featureSpace - A string in ('linear', quadratic', gaussian')
# Output:
#   feature vector - The features for the model free component
@njit
def _get_features(states, feature_space):
    n = states.shape[0]
    if feature_space == 'linear':
        features = np.zeros((n, 3))
        for i in range(n):
            features[i, :] = _get_feature_space_linear(states[i, :])
    else:
        features = np.zeros((n, 6))
        for i in range(n):
            features[i, :] = _get_feature_space_quadratic(states[i, :])
    return features


# A class for returning the featureSpace for the model-free estimate (V-learning)
# Attributes:
#   featureSpace - A string in ('linear', quadratic', gaussian')
# Methods:   
#   __init__
#   get_features_current(data)
#   get_features_next(data)
class FeatureModel:
    def __init__(self, feature_space='quadratic'):
        self.s2max = None
        self.s2min = None
        self.s1max = None
        self.s1min = None
        if feature_space not in ['quadratic', 'linear', 'gaussian']:
            print("The variable featureSpace must be 'quadratic', 'linear', or 'gaussian'")
            print("Setting featureSpace to the default value of 'quadratic'")
            feature_space = 'quadratic'
        self.featureSpace = feature_space
        if self.featureSpace == 'gaussian':
            self.s1Min = None
            self.s1Max = None
            self.s2Min = None
            self.s2Max = None

    def set_gaussian_params(self, data):
        states = data.get_states()
        self.s1min = np.min(states[:, 0])
        self.s1max = np.max(states[:, 0])
        self.s2min = np.min(states[:, 1])
        self.s2max = np.max(states[:, 1])

    # get_features_current(data) - Returns the features for the current states
    # Input:
    #   data - An object of class DataObject
    # Output:
    #   features - A 2d numpy array containing the features for the current states
    def get_features_current(self, data):
        if self.featureSpace == 'gaussian':
            return _get_feature_space_gaussian(data.get_states(), self.s1max, self.s1min, self.s2max, self.s2min)
        else:
            return _get_features(data.get_states(), self.featureSpace)

    # get_features_next(data) - Returns the features for the next states
    # Input:
    #   data - An object of class DataObject
    # Output:
    #   features - A 2d numpy array containing the features for the next states
    def get_features_next(self, data):
        if self.featureSpace == 'gaussian':
            return _get_feature_space_gaussian(data.get_next_states(), self.s1max, self.s1min, self.s2max, self.s2min)
        else:
            return _get_features(data.get_next_states(), self.featureSpace)

    # get_features(data) - Returns the features for the provided states
    # Input:
    #   states - A 2d numpy array containing state vectors
    # Output:
    #   features - A 2d numpy array containing the features for the next states
    def get_features(self, states):
        if self.featureSpace == 'gaussian':
            return _get_feature_space_gaussian(states, self.s1max, self.s1min, self.s2max, self.s2min)
        else:
            return _get_features(states, self.featureSpace)
