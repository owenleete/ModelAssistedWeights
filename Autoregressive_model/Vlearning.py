import numpy as np
from DataObject import DataObject
from random import sample
import scipy.optimize as spo


# Computes value of parameter beta using MP inverse to avoid errors when using singular matrix
def la_beta(phi_current, phi_next, psi, probs, probs_gen, reward, discount, n_sub, pen_val=0.0):
    prob_vec = probs / probs_gen

    c = psi.T.dot((prob_vec[:, np.newaxis] * (discount * phi_next - phi_current))) / n_sub
    a = psi.T.dot((reward * prob_vec)[:, np.newaxis]) / n_sub

    temp = np.linalg.pinv(np.matmul(c.T, c) + pen_val * np.eye(c.shape[0]))

    beta = np.matmul(temp, -np.matmul(c.T, a))

    return np.squeeze(beta)


# Gets lists of indices for cross validation
def get_index_list(n_splits, n_sub):
    index_list = np.array(sample((list(range(n_splits)) * int(np.ceil(n_sub / n_splits)))[:n_sub], n_sub))
    return index_list


# split data for cross validation
def split_data(data, index_list, psi, index):
    train_data = DataObject(data.state_dim)
    test_data = DataObject(data.state_dim)
    train_data.set_data(data.data[np.in1d(data.data[:, 0], np.where(index_list != index)), :],
                        len(np.where(index_list != index)[0]), data.n_obs)
    test_data.set_data(data.data[np.in1d(data.data[:, 0], np.where(index_list == index)), :],
                       len(np.where(index_list == index)[0]), data.n_obs)
    train_psi = psi[np.in1d(data.data[:, 0], np.where(index_list != index)), :]
    test_psi = psi[np.in1d(data.data[:, 0], np.where(index_list == index)), :]
    return train_data, test_data, train_psi, test_psi


# Calculate beta for V-learning model
def get_beta_v_learn(data_object, psi, policy_object, feature_model, discount, pen_val=0.0):
    states = data_object.get_states()
    beta = la_beta(feature_model.get_features_current(data_object), feature_model.get_features_next(data_object), psi,
                   policy_object.get_probs(states, data_object.get_actions()), data_object.get_probs(),
                   data_object.get_rewards(), discount, data_object.n_sub, pen_val)
    return beta


# Calculate temporal difference error for a given value of beta
def td_error(beta, data, feature_model, policy_eval, discount):
    val_current = feature_model.get_features_current(data).dot(beta)
    val_next = feature_model.get_features_next(data).dot(beta)
    probs_gen = data.get_probs()
    probs = policy_eval.get_probs(data.get_states(), data.get_actions())
    reward = data.get_rewards()
    return ((probs / probs_gen) * (reward + discount * val_next - val_current)).sum(axis=0) / data.n_sub


# Selects best value for penalization based on cross validation
def select_tuning_parameter(lambda_vec, data_object, psi, policy_eval, feature_model, discount, n_splits=5,
                            n_rounds=5):
    lambda_res_full = np.zeros_like(lambda_vec)
    for k in range(n_rounds):
        index_list = get_index_list(n_splits, data_object.n_sub)
        lambda_res = np.zeros_like(lambda_vec)
        for i in range(n_splits):
            train_data, test_data, train_psi, _ = split_data(data_object, index_list, psi, i)
            for j in range(lambda_vec.size):
                beta = get_beta_v_learn(train_data, train_psi, policy_eval, feature_model, discount, lambda_vec[j])
                lambda_res[j] += td_error(beta, test_data, feature_model, policy_eval, discount)
        lambda_res_full += np.abs(lambda_res)
    return lambda_vec[np.argmin(np.abs(lambda_res_full))]


# Fits the V full-learning model, including tuning parameter selection
def fit_v_learning(data_object, policy_eval, feature_model, discount, lambda_vec, n_splits=5, n_rounds=5, psi=None):
    if psi is None:
        psi = feature_model.get_features_current(data_object)
    pen_val = select_tuning_parameter(lambda_vec, data_object, psi, policy_eval, feature_model, discount, n_splits,
                                      n_rounds)
    beta = get_beta_v_learn(data_object, psi, policy_eval, feature_model, discount, pen_val)
    return beta, pen_val


# Get model for numerator of modeled Godambe weights
def get_num(data, policy_eval, feature_model, discount):
    probs_gen = data.get_probs()
    probs = policy_eval.get_probs(data.get_states(), data.get_actions())
    phi = feature_model.get_features_current(data)
    phi_next = feature_model.get_features_next(data)
    target = (probs/probs_gen)[:, np.newaxis]*(discount*phi_next - phi)
    rhs = np.einsum('ki,kj->ij', phi, phi)
    lhs = np.einsum('ki,kj->ij', target, phi)
    params = np.matmul(lhs, np.linalg.inv(rhs))
    return np.matmul(phi, params.T)


# Get model for denominator of modeled Godambe weights
def get_den(data, beta, policy_eval, feature_model, discount):
    zeta = np.zeros_like(beta)
    probs_gen = data.get_probs()
    probs = policy_eval.get_probs(data.get_states(), data.get_actions())
    phi = feature_model.get_features_current(data)
    phi_next = feature_model.get_features_next(data)
    target = np.log(((probs / probs_gen) * (discount * np.matmul(phi_next, beta) - np.matmul(phi, beta)))**2 + 1e-300)

    def func(param):
        return np.sum(np.abs(np.sum((target - np.matmul(phi, param))[:, np.newaxis] *
                      phi, axis=0)))

    temp = spo.minimize(func, zeta, method='Nelder-Mead')

    zeta = temp.x

    lp = np.minimum(np.matmul(phi, zeta), 700)

    return np.exp(lp)


# Fit V-learning using a model for the Godambe weights that is estimated from data
def fit_godambe_modeled(data_object, policy_eval, feature_model, discount):
    num = get_num(data_object, policy_eval, feature_model, discount)
    psi = feature_model.get_features_current(data_object)
    beta = np.zeros_like(psi[0])
    beta_old = np.zeros_like(psi[0])
    diff = 1
    t = 0
    n = 1
    learning_rate = 0.9
    while diff > 1e-6:
        beta = get_beta_v_learn(data_object, psi, policy_eval, feature_model, discount, 1e-5)
        den = get_den(data_object, beta, policy_eval, feature_model, discount)

        den = den+1e-2

        psi = (1-learning_rate**n)*psi + (learning_rate**n)*(num/den[:, np.newaxis])
        diff = np.sum(np.abs(beta-beta_old))
        beta_old = beta.copy()
        if t > 200:
            print("didn't converge")
            break
        t += 1
        n += 1
    return beta
