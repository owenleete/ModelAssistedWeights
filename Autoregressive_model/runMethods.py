import numpy as np
from Vlearning import fit_v_learning, fit_godambe_modeled
from pyGetValues import get_values, get_weights, get_psi
from DataObject import DataObject

# Dimension of the state vector
STATE_DIM = 2


class VLearningParameters:
    def __init__(self, lambda_vec=np.array([1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06, 1.e-07, 1.e-08]),
                 n_splits=5, n_rounds=5):
        assert type(lambda_vec) is np.ndarray, 'lambda_vec must be a numpy array'
        assert lambda_vec.ndim == 1, 'lambda_vec must be a 1-dimensional numpy array'
        self.lambda_vec = lambda_vec
        try:
            self.n_splits = int(n_splits)
        except TypeError:
            assert False, 'n_splits must be able to be casted to integer'
        try:
            self.n_rounds = int(n_rounds)
        except TypeError:
            assert False, 'n_rounds must be able to be casted to integer'


class DRParameters:
    def __init__(self, n_states_values=10000, mc_reps_values=1000, trajectory_length=100,
                 n_subjects_psi=10000, mc_reps_psi=5000, n_reps_alpha=10):
        self.n_states_values = n_states_values
        self.mc_reps_values = mc_reps_values
        self.trajectory_length = trajectory_length
        self.n_subjects_psi = n_subjects_psi
        self.mc_reps_psi = mc_reps_psi
        self.n_reps_alpha = n_reps_alpha


def run_vl(data_object, policy_eval, feature_model, v_learn_param, discount, ref_dist):
    if feature_model.featureSpace == 'gaussian':
        feature_model.set_gaussian_params(data_object)
    beta, _ = fit_v_learning(data_object, policy_eval, feature_model, discount, v_learn_param.lambda_vec,
                             v_learn_param.n_splits, v_learn_param.n_rounds)
    model_free_values = feature_model.get_features(ref_dist).dot(beta)
    return np.mean(model_free_values)


def run_ma(data_object, policy_gen, policy_eval, feature_model, trans_model, v_learn_param, dr_param, discount,
           ref_dist, max_procs):
    if feature_model.featureSpace == 'gaussian':
        feature_model.set_gaussian_params(data_object)

    states_train = ref_dist.copy()
    values_train = get_values(states_train, policy_eval, trans_model, dr_param.trajectory_length,
                              dr_param.mc_reps_values, discount, max_procs)

    phi_train = feature_model.get_features(states_train)

    beta = np.linalg.lstsq(phi_train, values_train, rcond=None)[0]
    fs = get_weights(data_object.get_states(), beta, discount, policy_eval, trans_model, dr_param.mc_reps_psi,
                     feature_model.featureSpace, max_procs)

    psi_g = get_psi(data_object.get_states(), beta, discount, policy_gen, policy_eval, trans_model,
                    dr_param.mc_reps_psi, feature_model.featureSpace, max_procs)

    new_psi = psi_g - np.mean(psi_g * fs[:, np.newaxis], axis=0) / np.mean(fs ** 2 + 0.001) * fs[:, np.newaxis]
    beta_new, _ = fit_v_learning(data_object, policy_eval, feature_model, discount, v_learn_param.lambda_vec,
                                 v_learn_param.n_splits, v_learn_param.n_rounds, new_psi)

    model_free_values = feature_model.get_features(ref_dist).dot(beta_new)
    return np.mean(model_free_values)


def run_mb(policy_eval, trans_model, dr_param, discount, ref_dist, max_procs):
    values = get_values(ref_dist, policy_eval, trans_model, dr_param.trajectory_length, dr_param.mc_reps_values,
                        discount, max_procs)
    return np.mean(values)


def run_aug(data, policy_gen, policy_eval, trans_model, feature_model, v_learn_param, ref_dist, discount):
    if feature_model.featureSpace == 'gaussian':
        feature_model.set_gaussian_params(data)
    data2 = DataObject(STATE_DIM, data.randomSeed)
    data2.generate(data.n_sub, data.n_obs, policy_gen, trans_model)

    data2.data = np.vstack((data2.data, data.data))
    data2.n_sub = data.n_sub + data2.n_sub

    beta, _ = fit_v_learning(data2, policy_eval, feature_model, discount, v_learn_param.lambda_vec,
                             v_learn_param.n_splits, v_learn_param.n_rounds)
    model_free_values = feature_model.get_features(ref_dist).dot(beta)
    return np.mean(model_free_values)


def run_gm(data_object, policy_eval, feature_model, discount, ref_dist):
    if feature_model.featureSpace == 'gaussian':
        feature_model.set_gaussian_params(data_object)
    beta = fit_godambe_modeled(data_object, policy_eval, feature_model, discount)
    model_free_values = feature_model.get_features(ref_dist).dot(beta)
    return np.mean(model_free_values)


def run_vg(data_object, policy_gen, policy_eval, feature_model, trans_model, v_learn_param, dr_param, discount,
           ref_dist, max_procs):
    if feature_model.featureSpace == 'gaussian':
        feature_model.set_gaussian_params(data_object)

    data_new = DataObject(data_object.state_dim, data_object.randomSeed + 1)
    data_new.generate(dr_param.n_subjects_psi, data_object.n_obs, policy_gen, trans_model)

    beta, _ = fit_v_learning(data_new, policy_eval, feature_model, discount, v_learn_param.lambda_vec,
                             v_learn_param.n_splits, v_learn_param.n_rounds)

    psi_g = get_psi(data_object.get_states(), beta, discount, policy_gen, policy_eval, trans_model,
                    dr_param.mc_reps_psi, feature_model.featureSpace, max_procs)

    beta_new, _ = fit_v_learning(data_object, policy_eval, feature_model, discount, v_learn_param.lambda_vec,
                                 v_learn_param.n_splits, v_learn_param.n_rounds, psi_g)

    model_free_values = feature_model.get_features(ref_dist).dot(beta_new)
    return np.mean(model_free_values)
