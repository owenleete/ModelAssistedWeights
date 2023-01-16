import os
import numpy as np
from PolicyObject import PolicyObject
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from functools import partial
from boundsReadWrite import read_bounds_transformer, write_bounds_transformer, set_p_bounds, file_len


def optim_wrapper(zeta0, zeta1, zeta2, zeta3, model_object, method, penalty=0.0002):
    policy_eval = PolicyObject(np.array([zeta0, zeta1, zeta2, zeta3]))
    value = model_object.fit_model(method, policy_eval)
    return value - penalty * np.sum(np.array([zeta0, zeta1, zeta2, zeta3]) ** 2)


def bayes_opt(sim_dir, filename, model_object, method, init_points=30, search_points=30, bound_points=40,
              p_bounds=None):
    opt_func = partial(optim_wrapper, model_object=model_object, method=method)

    if p_bounds is None:
        p_bounds = {'zeta0': (-50, 50), 'zeta1': (-50, 50), 'zeta2': (-50, 50), 'zeta3': (-50, 50)}

    # noinspection PyTypeChecker
    optimizer = BayesianOptimization(
        f=opt_func,
        pbounds=p_bounds,
        verbose=2,
    )

    logfile = sim_dir + 'logs/' + filename + '_logs.json'
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    os.makedirs(os.path.dirname(sim_dir + 'bounds/temp.txt'), exist_ok=True)

    try:
        load_logs(optimizer, logs=[logfile])
        runs_complete = file_len(logfile)
    except FileNotFoundError:
        runs_complete = 0

    if runs_complete != 0:
        if runs_complete > init_points + search_points:
            bound_points = bound_points + search_points + init_points - runs_complete
            search_points = 0
            init_points = 0
        elif runs_complete > init_points:
            search_points = search_points + init_points - runs_complete
            init_points = 0
        else:
            init_points = init_points - runs_complete

    if bound_points + search_points + init_points > 0:
        logger = JSONLogger(path=logfile, reset=False)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        if init_points > 0:
            for K in range(init_points):
                optimizer.maximize(
                    init_points=1,
                    n_iter=0,
                    acq="poi",
                    xi=0.0001,
                )

        if search_points > 0:
            for K in range(search_points):
                optimizer.maximize(
                    init_points=0,
                    n_iter=1,
                    acq="poi",
                    xi=0.0001,
                )

        if bound_points > 0:
            try:
                bounds_transformer = read_bounds_transformer(sim_dir + 'bounds/' + filename)
                p_bounds = set_p_bounds(bounds_transformer)
            except FileNotFoundError:
                bounds_transformer = SequentialDomainReductionTransformer(gamma_osc=0.9, gamma_pan=1.05, eta=0.95)
                p_bounds = {'zeta0': (-50, 50), 'zeta1': (-50, 50), 'zeta2': (-50, 50), 'zeta3': (-50, 50)}

            # noinspection PyTypeChecker
            optimizer = BayesianOptimization(
                f=opt_func,
                pbounds=p_bounds,
                verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                bounds_transformer=bounds_transformer,
            )

            load_logs(optimizer, logs=[logfile])
            # noinspection PyUnusedLocal
            runs_complete = file_len(logfile)

            logger = JSONLogger(path=logfile, reset=False)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

            for K in range(bound_points):
                optimizer.maximize(
                    init_points=0,
                    n_iter=1,
                    acq="poi",
                    xi=0.0001,
                )

                write_bounds_transformer(sim_dir + 'bounds/' + filename, bounds_transformer)

    zeta = np.zeros(4)

    zeta[0] = optimizer.max['params']['zeta0']
    zeta[1] = optimizer.max['params']['zeta1']
    zeta[2] = optimizer.max['params']['zeta2']
    zeta[3] = optimizer.max['params']['zeta3']

    return zeta
