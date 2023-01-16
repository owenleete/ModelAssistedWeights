import numpy as np
from DataObject import DataObject
from random import sample


class VLearningParameters:
    def __init__(self, pen_vec=np.array([1 * 10 ** (-j) for j in range(1, 9)]), n_splits=10, n_rounds=5):
        """
        Initialize object for V-learning hyper parameters

        Parameters
        ----------
        pen_vec : 1d numpy array, optional
            Different penalization values for V-learning. The default is np.array([1*10**(-j) for j in range(1,9)]).
        n_splits : int, optional
            Number of splits for k-fold cross validation. The default is 10.
        n_rounds : int, optional
            Number of rounds of k-fold cross validation. The default is 5.

        Returns
        -------
        None.

        """
        assert type(pen_vec) is np.ndarray, 'penVec must be a numpy array'
        assert pen_vec.ndim == 1, 'penVec must be a 1-dimensional numpy array'
        self.penVec = pen_vec
        try:
            self.nSplits = int(n_splits)
        except:
            assert False, 'nSplits must be castable to integer'
        try:
            self.nRounds = int(n_rounds)
        except:
            assert False, 'nRounds must be castable to integer'


def la_beta(features_current, features_next, ee_weights, probs, probs_gen, reward, discount, n_sub, pen_val=0.0):
    """
    Use linear algebra formulation to calculate V-learning parameter 'beta'

    Parameters
    ----------
    features_current : 2d numpy array
        Features of current states.
    features_next : 2d numpy array
        Features of next states.
    ee_weights : 2d numpy array
        Estimating equation weights.
    probs : 1d numpy array
        Probability of selecting action under evaluation policy.
    probs_gen : 1d numpy array
        Probability of selecting action under generating policy.
    reward : 1d numpy array
        Array of rewards.
    discount : float
        Discount factor gamma.
    n_sub : int
        Number of subjects in the data.
    pen_val : float, optional
        L1 penalty on the parameters. The default is 0.0.

    Returns
    -------
    1d numpy array
        Value of beta.

    """

    prob_vec = probs / probs_gen

    c = ee_weights.T.dot((prob_vec[:, np.newaxis] * (discount * features_next - features_current))) / n_sub
    a = ee_weights.T.dot((reward * prob_vec)[:, np.newaxis]) / n_sub

    try:
        temp = np.linalg.inv(np.matmul(c.T, c) + pen_val * np.eye(c.shape[0]))
    except:
        temp = np.linalg.pinv(np.matmul(c.T, c) + pen_val * np.eye(c.shape[0]))

    beta = np.matmul(temp, -np.matmul(c.T, a))

    return np.squeeze(beta)


def get_index_list(n_splits, n_sub):
    """
    Get a list of random splits of the data

    Parameters
    ----------
    n_splits : int
        The number of splits "k" for k-fold validation.
    n_sub : int
        The number of subjects in the data.

    Returns
    -------
    index_list : 1d numpy array
        Array of ints corresponding to random splits of the data.

    """
    index_list = np.array(sample((list(range(n_splits)) * int(np.ceil(n_sub / n_splits)))[:n_sub], n_sub))
    return index_list


def split_data(data_object, index_list, ee_weights, index):
    """
    Split data and estimating equation weights into training and testing data sets.

    Parameters
    ----------
    data_object : DataObject
        Object containing the data.
    index_list : 1d numpy array
        random splits of the data.
    ee_weights : 2d numpy array
        estimating equation weights.
    index : int
        Which index to split on.

    Returns
    -------
    train_data : DataObject
        Training data.
    test_data : DataObject
        Testing data.
    train_ee_weights : 2d Numpy array
        training estimating equation weights.
    test_ee_weights : 2d Numpy array
        testing estimating equation weights.

    """
    train_data = DataObject(data_object.stateDim)
    test_data = DataObject(data_object.stateDim)
    train_data.set_data(data_object.data[np.in1d(data_object.data[:, 0], np.where(index_list != index)), :],
                        len(np.where(index_list != index)[0]), data_object.n_obs)
    test_data.set_data(data_object.data[np.in1d(data_object.data[:, 0], np.where(index_list == index)), :],
                       len(np.where(index_list == index)[0]), data_object.n_obs)
    train_ee_weights = ee_weights[np.in1d(data_object.data[:, 0], np.where(index_list != index)), :]
    test_ee_weights = ee_weights[np.in1d(data_object.data[:, 0], np.where(index_list == index)), :]
    return train_data, test_data, train_ee_weights, test_ee_weights


def get_beta_v_learn(data_object, ee_weights, policy_object, feature_model, discount, pen_val=0.0):
    """
    Get beta for V-learning 

    Parameters
    ----------
    data_object : DataObject
        Object containing the data.
    ee_weights : 2d numpy array
        estimating equation weights.
    policy_object : PolicyObject
        Evaluation policy.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    pen_val : float, optional
        L1 penalty on the parameters. The default is 0.0.

    Returns
    -------
    beta : 1d numpy array
        Value of beta.

    """
    states = data_object.get_states()
    beta = la_beta(feature_model.get_features_current(data_object), feature_model.get_features_next(data_object),
                   ee_weights, policy_object.get_probs(states, data_object.get_actions()), data_object.get_probs(),
                   data_object.get_rewards(), discount, data_object.n_sub, pen_val)
    return beta


def td_error(beta, data_object, feature_model, policy_eval, discount):
    """
    Get temporal difference error for the supplied value of beta

    Parameters
    ----------
    beta : 1d numpy array
        Parameter values for V-learning.
    data_object : DataObject
        Object containing the data.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    policy_eval : PolicyObject
        Evaluation policy.
    discount : float
        Discount factor gamma.

    Returns
    -------
    float
        Temporal difference error.

    """
    val_current = feature_model.get_features_current(data_object).dot(beta)
    val_next = feature_model.get_features_current(data_object).dot(beta)
    probs_gen = data_object.get_probs()
    probs = policy_eval.get_probs(data_object.get_states(), data_object.get_actions())
    reward = data_object.get_rewards()
    return ((probs / probs_gen) * (reward + discount * val_next - val_current)).sum(axis=0) / data_object.n_sub


def select_tuning_parameter(pen_vec, data_object, ee_weights, policy_eval, feature_model, discount, n_splits=5,
                            n_rounds=5):
    """
    Find the best value for the penalty term with k-fold cross validation

    Parameters
    ----------
    pen_vec : 1d numpy array
            Different penalization values for V-learning.
    data_object : DataObject
        Object containing the data.
    ee_weights : 2d numpy array
        estimating equation weights.
    policy_eval : PolicyObject
        Evaluation policy.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    n_splits : int, optional
        The number of splits "k" for k-fold validation.
    n_rounds : int, optional
        Number of rounds of k-fold cross validation. The default is 5.

    Returns
    -------
    float
        The optimal penalty value to minimize k-fold TD error.

    """
    lambda_res_full = np.zeros_like(pen_vec)
    for k in range(n_rounds):
        index_list = get_index_list(n_splits, data_object.n_sub)
        lambda_res = np.zeros_like(pen_vec)
        for i in range(n_splits):
            train_data, test_data, train_ee_weights, _ = split_data(data_object, index_list, ee_weights, i)
            for j in range(pen_vec.size):
                beta = get_beta_v_learn(train_data, train_ee_weights, policy_eval, feature_model, discount, pen_vec[j])
                lambda_res[j] += td_error(beta, test_data, feature_model, policy_eval, discount)
        lambda_res_full += np.abs(lambda_res)
    return pen_vec[np.argmin(np.abs(lambda_res_full))]


def fit_v_learning(data_object, policy_eval, feature_model, discount, v_params, ee_weights=None):
    """
    Find the optimal penalty value and calculate V-learning solution

    Parameters
    ----------
    data_object : DataObject
        Object containing the data.
    policy_eval : PolicyObject
        Evaluation policy.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    v_params : VLearningParameters
        V-learning hyperparameter object.
    ee_weights : 2d numpy array, optional
        Estimating equation weights. The default is None.

    Returns
    -------
    beta : 1d numpy array
        Parameter solution to V-learning approach.
    pen_val : float
        The optimal penalty value to minimize k-fold TD error.

    """
    if ee_weights is None:
        ee_weights = feature_model.get_features_current(data_object)
    pen_val = select_tuning_parameter(v_params.penVec, data_object, ee_weights, policy_eval, feature_model, discount,
                                      v_params.nSplits, v_params.nRounds)
    beta = get_beta_v_learn(data_object, ee_weights, policy_eval, feature_model, discount, pen_val)
    return beta, pen_val
