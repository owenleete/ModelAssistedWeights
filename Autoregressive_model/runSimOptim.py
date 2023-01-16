import os
import pdb
import sys
import numpy as np
from PolicyObject import PolicyObject
from TransitionModel import TransitionModel
from FeatureModel import FeatureModel
from DataObject import DataObject
from modelObject import ModelObject
from runMethods import VLearningParameters, DRParameters
from runOptim import bayes_opt

algo = sys.argv[3]
N = int(sys.argv[1])
T = int(sys.argv[2])
j = int(os.environ['SLURM_ARRAY_TASK_ID'])
max_procs = int(os.environ['SLURM_CPUS_PER_TASK'])
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

# create baseline policy
policy_eval = PolicyObject(np.array([0., 0., 0., 0.]))

# Create object for the model components
# this will contain everything necessary to run every approach considered
model_object = ModelObject(data, policy_eval, policy_gen, trans_model, feature_model, dr_param, v_learn_param,
                           ref_dist, discount, max_procs)

sim_dir = os.getcwd()+'/policy_optim/'
filename = algo + basis1 + str(N).zfill(3) + str(T).zfill(3) + str(j).zfill(4)


def save_result(sim_dir_, file_name, policy, model_gen_):
    os.makedirs(os.path.dirname(sim_dir_ + 'results/values/' + file_name + '.csv'), exist_ok=True)

    file_name_ = sim_dir_ + 'results/' + file_name + '.csv'
    # noinspection PyTypeChecker
    np.savetxt(file_name_, np.reshape(policy, (1, 4)), delimiter=',')

    value = model_gen_.fit_model('MB')

    file_name_ = sim_dir_ + 'results/values/' + file_name + '.csv'
    # noinspection PyTypeChecker
    np.savetxt(file_name_, np.array([value]), delimiter=',')


# Can change 'MA' to any of the above approach abbreviations
# need to increase number of init/search/bound points for real run (kept low for example code)
policy_params = bayes_opt(sim_dir, filename, model_object, algo, init_points=300, search_points=600, bound_points=100)

policy_est = PolicyObject(policy_params)

# create model_object with true system dynamics
model_gen = ModelObject(data, policy_est, policy_gen, trans_gen, feature_model, dr_param, v_learn_param,
                        ref_dist, discount, max_procs)

save_result(sim_dir, filename, policy_params, model_gen)
