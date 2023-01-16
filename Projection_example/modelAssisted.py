import numpy as np
from scipy.optimize import linprog
from Vlearning import fit_v_learning


def set_values(t_params, p_params, reward_vec, discount):
    """
    Calculate the state-values [see Puterman (2005) or Sutton & Barto (2020) for definition of value] for each of the
    states

    Parameters
    ----------
    t_params : 3d numpy array
        Parameters governing transition dynamics.
    p_params : 1d numpy array
        Parameters governing action selection policy.
    reward_vec : 1d numpy array
           Vector containing reward values
    discount : float
        Discount factor gamma.

    Returns
    -------
    1d numpy array
        Array of values for each state.

    """
    params = np.zeros_like(t_params[0, :, :])
    avg_reward = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        lp = min(500, p_params[i])
        prob1 = np.exp(lp) / (1 + np.exp(lp))
        prob0 = 1 - prob1
        params[i, :] = prob0 * t_params[0, i, :] + prob1 * t_params[1, i, :]
        reward_vec0 = reward_vec
        reward_vec1 = reward_vec - 0.5
        avg_reward[i] = np.sum(prob0 * reward_vec0 * t_params[0, i, :] + prob1 * reward_vec1 * t_params[1, i, :])
    aub = -(np.eye(params.shape[0]) - discount * params)
    bub = -avg_reward
    c = [1.0] * params.shape[0]
    res = linprog(c, aub, bub, np.zeros_like(aub), np.zeros_like(bub), (-np.Inf, np.Inf))
    return res.x


def get_projection(data_object, policy_eval, trans_model, feature_model, discount):
    """
    Calculate the projection of the state-values onto the feature space

    Parameters
    ----------
    data_object : DataObject
        Object containing the data.
    trans_model : TransitionModel
        Transition dynamics model object for the MDP.
    policy_eval : PolicyObject
        Evaluation policy.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.

    Returns
    -------
    beta : 1d numpy array
        Parameter solution to the projection of the values on the feature space.

    """
    values = set_values(trans_model.get_parameters(), policy_eval.params, trans_model.reward_vec, discount)
    states = np.unique(data_object.data[:, 2])
    y = np.zeros(states.shape[0])
    for i in range(states.shape[0]):
        y[i] = values[int(states[i])]
    x = feature_model.get_features(states[:, np.newaxis])
    # Calculate projection
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta


def get_e_tde(data_object, policy_eval, trans_model, feature_model, discount):
    """
    Get expected temporal difference error and feature space for each of the states.

    Parameters
    ----------
    data_object : DataObject
        Object containing the data.
    policy_eval : PolicyObject
        Evaluation policy.
    trans_model : TransitionModel
        Transition dynamics model object for the MDP.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.

    Returns
    -------
    e_tde : 2d numpy array
        Expected TD error for each state.
    features : 2d numpy array
        Feature space for each state.

    """
    beta_proj = get_projection(data_object, policy_eval, trans_model, feature_model, discount)
    states = np.array([[x] for x in np.sort(np.unique(data_object.get_states()))])
    e_tde = np.zeros(states.size)
    for i in range(states.size):
        state = states[i]
        prob = policy_eval.get_prob(state, 1)
        e_tde[i] = (np.sum((1 - prob) * trans_model.get_parameters()[0, int(state[0]), :] * (
                    trans_model.reward_vec + discount * feature_model.get_features(states).dot(beta_proj) -
                    feature_model.get_feature(state).dot(beta_proj))) +
                    np.sum(prob * trans_model.get_parameters()[1, int(state[0]), :] *
                           ((trans_model.reward_vec - 0.5) + discount * feature_model.get_features(states).dot(
                               beta_proj) - feature_model.get_feature(state).dot(beta_proj))))
    features = feature_model.get_features(states)
    return e_tde, features


def get_ma_weights(data_object, policy_eval, policy_gen, trans_model, feature_model, discount):
    """
    Calculate the model assisted weights

    Parameters
    ----------
    data_object : DataObject
        Object containing the data.
    policy_eval : PolicyObject
        Evaluation policy.
    policy_gen : PolicyObject
        Generating/behaviour/logging policy.
    trans_model : TransitionModel
        Transition dynamics model object for the MDP.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.

    Returns
    -------
    ma_weights : 2d numpy array
        Model assisted estimating equation weights.

    """
    e_tde, features = get_e_tde(data_object, policy_eval, trans_model, feature_model, discount)
    features_vec = np.zeros((data_object.data.shape[0], 2))
    e_tde_vec = np.zeros(data_object.data.shape[0])
    for i in range(data_object.data.shape[0]):
        features_vec[i, :] = features[int(data_object.data[i, 2]), :]
        e_tde_vec[i] = e_tde[int(data_object.data[i, 2])]
    probs_eval = policy_eval.get_probs(data_object.get_states(), data_object.get_actions())
    probs_gen = policy_gen.get_probs(data_object.get_states(), data_object.get_actions())

    # Correction Factor for weights
    corr_fact = np.sum(((probs_eval / probs_gen) * e_tde_vec)[:, np.newaxis] * features_vec, axis=0) / np.sum(
        ((probs_eval / probs_gen) * e_tde_vec ** 2))

    # Calculate model assisted weights
    ma_weights = features_vec - e_tde_vec[:, np.newaxis] * corr_fact[np.newaxis, :]

    return ma_weights


def fit_ma(data_object, policy_eval, policy_gen, trans_model, feature_model, discount, v_params):
    """
    Fit the model assisted weight method.

    Parameters
    ----------
    data_object : DataObject
        Object containing the data.
    policy_eval : PolicyObject
        Evaluation policy.
    policy_gen : PolicyObject
        Generating/behaviour/logging policy.
    trans_model : TransitionModel
        Transition dynamics model object for the MDP.
    feature_model : FeatureModel
        Model for the V-learning feature space.
    v_params : VLearningParameters
        V-learning hyperparameter object.
    discount : float
        Discount factor gamma.

    Returns
    -------
    beta_ma : 1d numpy array
        Parameter solution to model assisted approach.

    """
    ma_weights = get_ma_weights(data_object, policy_eval, policy_gen, trans_model, feature_model, discount)
    beta_ma, _ = fit_v_learning(data_object, policy_eval, feature_model, discount, v_params, ee_weights=ma_weights)
    return beta_ma
