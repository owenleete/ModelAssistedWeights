import numpy as np
from bayes_opt import SequentialDomainReductionTransformer
import yaml
import io


def write_bounds_transformer(file_name, bounds_transformer):
    bounds_object = {}
    for i in range(len(bounds_transformer.bounds)):
        bounds_object[i] = bounds_transformer.bounds[i].tolist()

    bounds_transformer_object = {
        "bounds": bounds_object,
        "c": bounds_transformer.c.tolist(),
        "c_hat": bounds_transformer.c_hat.tolist(),
        "contraction_rate": bounds_transformer.contraction_rate.tolist(),
        "current_d": bounds_transformer.current_d.tolist(),
        "current_optimal": bounds_transformer.current_optimal.tolist(),
        "eta": bounds_transformer.eta,
        "gamma": bounds_transformer.gamma.tolist(),
        "gamma_osc": bounds_transformer.gamma_osc,
        "gamma_pan": bounds_transformer.gamma_pan,
        "original_bounds": bounds_transformer.original_bounds.tolist(),
        "previous_d": bounds_transformer.previous_d.tolist(),
        "previous_optimal": bounds_transformer.previous_optimal.tolist(),
        "r": bounds_transformer.r.tolist(),
    }

    filename1 = file_name + ".yaml"
    filename2 = file_name + "_backup.yaml"

    with io.open(filename1, 'w', encoding='utf8') as outfile:
        yaml.dump(bounds_transformer_object, outfile, default_flow_style=False, allow_unicode=True)
    with io.open(filename2, 'w', encoding='utf8') as outfile:
        yaml.dump(bounds_transformer_object, outfile, default_flow_style=False, allow_unicode=True)


def create_bounds_transformer(file_name):
    bounds_transformer = SequentialDomainReductionTransformer(gamma_osc=0.7, gamma_pan=1.0, eta=0.9)

    with open(file_name, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    bounds = []
    for bound in data_loaded["bounds"].items():
        bounds.append(np.array(bound[1]))

    bounds_transformer.bounds = bounds
    bounds_transformer.c = np.array(data_loaded["c"])
    bounds_transformer.c_hat = np.array(data_loaded["c_hat"])
    bounds_transformer.contraction_rate = np.array(data_loaded["contraction_rate"])
    bounds_transformer.current_d = np.array(data_loaded["current_d"])
    bounds_transformer.current_optimal = np.array(data_loaded["current_optimal"])
    bounds_transformer.eta = data_loaded["eta"]
    bounds_transformer.gamma = np.array(data_loaded["gamma"])
    bounds_transformer.gamma_osc = data_loaded["gamma_osc"]
    bounds_transformer.gamma_pan = data_loaded["gamma_pan"]
    bounds_transformer.original_bounds = np.array(data_loaded["original_bounds"])
    bounds_transformer.previous_d = np.array(data_loaded["previous_d"])
    bounds_transformer.previous_optimal = np.array(data_loaded["previous_optimal"])
    bounds_transformer.r = np.array(data_loaded["r"])

    if len(bounds_transformer.r) != len(bounds_transformer.c):
        raise Exception('Incomplete File')

    return bounds_transformer


def read_bounds_transformer(file_name):
    filename1 = file_name + ".yaml"
    filename2 = file_name + "_backup.yaml"
    try:
        bounds_transformer = create_bounds_transformer(filename1)
    except FileNotFoundError:
        bounds_transformer = create_bounds_transformer(filename2)
    return bounds_transformer


def set_p_bounds(bounds_transformer):
    bounds = bounds_transformer.bounds[-1]
    p_bounds = {'zeta0': (bounds[0, 0], bounds[0, 1]), 'zeta1': (bounds[1, 0], bounds[1, 1]),
                'zeta2': (bounds[2, 0], bounds[2, 1]), 'zeta3': (bounds[3, 0], bounds[3, 1])}
    return p_bounds


def file_len(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
