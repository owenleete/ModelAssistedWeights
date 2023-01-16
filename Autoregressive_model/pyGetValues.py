import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

wd = os.getcwd()


def convert_feature_space(feature_space):
    if feature_space == 'quadratic':
        feat_space = 1
    elif feature_space == 'gaussian':
        feat_space = 2
    elif feature_space == 'linear':
        feat_space = 3
    else:
        feat_space = 1
    return feat_space


getVal = ctypes.cdll.LoadLibrary(wd+"/getValues.so")


class GoSlice(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)),
                ("len", ctypes.c_longlong), ("cap", ctypes.c_longlong)]


getVal.PySetValues.argtypes = [GoSlice, GoSlice, GoSlice, GoSlice, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                               ctypes.c_int, ctypes.c_double, ctypes.c_int]
getVal.PyGetPsi.argtypes = [GoSlice, GoSlice, GoSlice, GoSlice, GoSlice, GoSlice, ctypes.c_double, ctypes.c_double,
                            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
getVal.PyGetWeights.argtypes = [GoSlice, GoSlice, GoSlice, GoSlice, GoSlice, ctypes.c_double, ctypes.c_double,
                                ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
getVal.PyGetPMWeights.argtypes = [GoSlice, GoSlice, GoSlice, GoSlice, GoSlice, GoSlice, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]


def string_out(vec):
    vec = np.reshape(vec, vec.size).tolist()
    return vec


# noinspection PyTypeChecker
def make_go_slice(vec):
    vec = string_out(vec)
    # noinspection PyArgumentList
    return GoSlice((ctypes.c_double * len(vec))(*vec), len(vec), len(vec))


def get_values(states, policy_object, transition_model, chain_length, num_chains, discount, max_procs):
    n = states.shape[0]
    theta1, theta2, sd1, sd2 = transition_model.get_parameters()
    p_params = policy_object.params
    getVal.PySetValues.restype = ndpointer(dtype=ctypes.c_double, shape=(n,))
    values = getVal.PySetValues(make_go_slice(states), make_go_slice(p_params), make_go_slice(theta1),
                                make_go_slice(theta2),
                                sd1, sd2, chain_length, num_chains, discount, max_procs)
    return values.copy()


def get_psi(states, beta, discount, policy_gen, policy_object, transition_model, n_states, feature_space, max_procs):
    feat_space = convert_feature_space(feature_space)
    n = states.shape[0]
    p = beta.size
    theta1, theta2, sd1, sd2 = transition_model.get_parameters()
    p_params_gen = policy_gen.params
    p_params = policy_object.params
    getVal.PyGetPsi.restype = ndpointer(dtype=ctypes.c_double, shape=(n * p,))
    psi_vec = getVal.PyGetPsi(make_go_slice(states), make_go_slice(beta), make_go_slice(p_params_gen),
                              make_go_slice(p_params), make_go_slice(theta1), make_go_slice(theta2), sd1, sd2,
                              n_states, feat_space, discount, max_procs)
    psi = np.reshape(psi_vec, (n, p))
    return psi.copy()


def get_weights(states, beta, discount, policy_object, transition_model, n_states, feature_space, max_procs):
    feat_space = convert_feature_space(feature_space)
    n = states.shape[0]
    theta1, theta2, sd1, sd2 = transition_model.get_parameters()
    p_params = policy_object.params
    getVal.PyGetWeights.restype = ndpointer(dtype=ctypes.c_double, shape=(n,))
    weights = getVal.PyGetWeights(make_go_slice(states), make_go_slice(beta), make_go_slice(p_params),
                                  make_go_slice(theta1), make_go_slice(theta2), sd1, sd2, n_states, feat_space,
                                  discount, max_procs)
    return weights.copy()


def get_pm_weights(states, beta, discount, policy_object, policy_gen, transition_model, n_states, feature_space,
                   max_procs):
    feat_space = convert_feature_space(feature_space)
    n = states.shape[0]
    theta1, theta2, sd1, sd2 = transition_model.get_parameters()
    p_params = policy_object.params
    p_params_gen = policy_gen.params
    getVal.PyGetPMWeights.restype = ndpointer(dtype=ctypes.c_double, shape=(n,))
    weights = getVal.PyGetPMWeights(make_go_slice(states), make_go_slice(beta), make_go_slice(p_params),
                                    make_go_slice(p_params_gen), make_go_slice(theta1), make_go_slice(theta2), sd1, sd2,
                                    n_states, feat_space, discount, max_procs)
    return weights.copy()
