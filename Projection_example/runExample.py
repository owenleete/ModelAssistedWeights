import sys
import numpy as np


# Import objects/functions to define MDP, selection policy, and V-learning method
from TransitionModel import TransitionModel
from PolicyObject import PolicyObject
from DataObject import DataObject
from FeatureModel import FeatureModel
from Vlearning import fit_v_learning, VLearningParameters
from modelAssisted import fit_ma, get_projection

if __name__ == '__main__':

    # Number of subjects
    try:
        n = int(sys.argv[1])
    except:
        n = 100
    # Number of observations per subject
    try:
        t = int(sys.argv[2])
    except:
        t = 100
    # Set random seed
    try:
        random_seed = int(sys.argv[3])
    except:
        random_seed = 1

    # Number of values needed to uniquely identify the state
    state_dimension = 1
    
    # MDP discount parameter
    discount = 0.9
    
    # Vector of rewards
    reward_vec = np.array([-0.5, 0.7, -0.25])
    
    # True system dynamics / data generating model
    trans_gen = TransitionModel(reward_vec, 3)
    # Define parameters for generating model
    t_params = np.zeros((2, 3, 3))
    t_params[0] = np.array([[0.05, 0.475, 0.475], [0.05, 0.85, 0.1], [0.15, 0.35, 0.5]])
    t_params[1] = np.array([[0.6, 0.0, 0.4], [0.3, 0.25, 0.45], [0.15, 0.8, 0.05]])
    # Set model parameters
    trans_gen.set_parameters(t_params)
    
    # Action selection model used to generate the data
    policy_gen = PolicyObject(np.zeros(3))
    
    # Proposed action selection model to evaluate
    policy_eval = PolicyObject(np.array([1., -1.0, -.25]))
    
    # Model for the feature space / basis functions for V-learning model
    feature_model = FeatureModel()
    
    # Set up data object
    data = DataObject(state_dimension, random_seed)
    # Generate data
    data.generate(n, t, policy_gen, trans_gen)
    
    # Set up transition model for estimated system dynamics
    trans_model = TransitionModel(reward_vec, 3)
    # Estimate system dynamics
    trans_model.fit_transition_model(data.data)
    
    # Set up hyper-parameters for V-learning
    #   Use default values for all
    v_params = VLearningParameters()
    
    # Fit standard V-learning
    beta_v, _ = fit_v_learning(data, policy_eval, feature_model, discount, v_params)
    # Fit V-learning with model assisted weights
    beta_ma = fit_ma(data, policy_eval, policy_gen, trans_model, feature_model, discount, v_params)
    
    # Get true projection value
    beta_proj = get_projection(data, policy_eval, trans_gen, feature_model, discount)
    
    print('Simulation with n=', n, ', t=', t, ', and random seed of ', random_seed, sep='')
    print('')
    print('Absolute error for beta_0:')
    print('Standard V-learning       ', round(abs(beta_proj[0] - beta_v[0]), 4))
    print('Model Assisted V-learning ', round(abs(beta_proj[0] - beta_ma[0]), 4))
    print('')
    print('Absolute error for beta_1:')
    print('Standard V-learning       ', round(abs(beta_proj[1] - beta_v[1]), 4))
    print('Model Assisted V-learning ', round(abs(beta_proj[1] - beta_ma[1]), 4))
    