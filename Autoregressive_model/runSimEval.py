import os
# import sys
import numpy as np
from PolicyObject import PolicyObject
from TransitionModel import TransitionModel
from FeatureModel import FeatureModel
from DataObject import DataObject
from modelObject import ModelObject
from runMethods import VLearningParameters, DRParameters
from runOptim import bayes_opt

#
N = 25            # number of subjects
T = 24            # number of decisions per subject
pol = 'eq'        # which policy to test 'eq' = equiprobable, 'op' = near optimal
j = 1             # random seed used to generate data
max_procs = 12    # number of available processors for parallelization
basis = 'linear'  # which feature space to use, 'linear', 'quadratic', or 'gaussian'
discount = 0.9    # discount factor gamma

basis1 = 'L'
if basis == 'linear':
    basis1 = 'L'
elif basis == 'quadratic':
    basis1 = 'Q'
elif basis == 'gaussian':
    basis1 = 'G'

# load data
ref_dist = np.load('initial.npy')[:, 0:2]
# set directory to save results to
simDir = os.getcwd()+'/policy_eval/'

# create objects for generating data (transition dynamics and generating policy)
policy_gen = PolicyObject(np.array([0., 0., 0., 0.]))
trans_gen = TransitionModel()
theta_1 = np.array([0, -0.75, 0, 0.25, 0, 1.5, 0])
theta_2 = np.array([0, 0, 0.75, 0.25, 0, 0, -1.5])
trans_gen.set_parameters(theta_1, theta_2, 0.25, 0.25)

# generate the data
data = DataObject(2, j)
data.generate(N, T, policy_gen, trans_gen)

# fit transition model
trans_model = TransitionModel()
trans_model.fit_transition_model(data)

# create object to translate state to feature vector
feature_model = FeatureModel(basis)

# parameters for v-learning and mc simulations (name is carryover from doubly robust version of method)
v_learn_param = VLearningParameters(lambda_vec=np.array([1.e-05, 1.e-06, 1.e-07, 1.e-08]))
dr_param = DRParameters(n_states_values=10000, mc_reps_values=2000, trajectory_length=100,
                        n_subjects_psi=10000, mc_reps_psi=2000, n_reps_alpha=25)

# create policy to evaluate
policy_eval = PolicyObject(np.array([0., 0., 0., 0.]))
if pol == 'eq':
    policy_eval = PolicyObject(np.array([0., 0., 0., 0.]))
elif pol == 'op':
    policy_eval = PolicyObject(np.array([-10., 50., -30., 10.]))

# Create object for the model components
# this will contain everything necessary to run every approach considered
model_object = ModelObject(data, policy_eval, policy_gen, trans_model, feature_model, dr_param, v_learn_param,
                           ref_dist, discount, max_procs)

# Run model-assisted approach
# (MA = model-assisted)
val_MA = model_object.fit_model('MA')
# Run model-based approach
# (MB = model-based)
val_MB = model_object.fit_model('MB')
# Run standard V-learning approach
# (VL = V-learning)
val_VL = model_object.fit_model('VL')
# Run V-learning + Godambe weights (estimated via MC)
# (VG = V-learning + godambe)
val_VG = model_object.fit_model('VG')
# Run V-learning + Godambe weights (estimated via parametric models)
# (GM = Godambe modeled)
val_GM = model_object.fit_model('GM')
# Run augmented data approach
# (AG = augmented)
val_AG = model_object.fit_model('AG')

sim_dir = os.getcwd()+'/policy_optim/'

# Can change 'MA' to any of the above approach abbreviations
# need to increase number of init/search/bound points for real run (kept low for example code)
policy_params = bayes_opt(sim_dir, 'test', model_object, 'MA', init_points=30, search_points=30, bound_points=40)

policy_est = PolicyObject(policy_params)

# create model_object with true system dynamics
model_gen = ModelObject(data, policy_est, policy_gen, trans_gen, feature_model, dr_param, v_learn_param,
                        ref_dist, discount, max_procs)

# Estimate of true value of estimated optimal policy
val_est = model_gen.fit_model('MB', policy_est)
