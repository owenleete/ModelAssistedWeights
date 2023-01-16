import sys
import numpy as np


# Import objects/functions to define MDP, selection policy, and V-learning method
from TransitionModel import TransitionModel
from PolicyObject import PolicyObject
from DataObject import DataObject
from FeatureModel import FeatureModel
from Vlearning import fit_v_learning, VLearningParameters
from modelAssisted import fit_ma, get_projection

def runSimulation(n):

    n_reps = 1000

    betaVec = np.zeros((n_reps, 2))
    betaVVec = np.zeros((n_reps, 2))
    betaMAVec = np.zeros((n_reps, 2))

    for l in range(n_reps):

        # Set follow-up period
        t = 100
        # Set random seed
        random_seed = l

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

        betaVec[l, :] = beta_proj
        betaVVec[l, :] = beta_v
        betaMAVec[l, :] = beta_ma

        print(l)

    # get "true" values of projection
    beta0 = np.mean(betaVec[:, 0])
    beta1 = np.mean(betaVec[:, 1])

    MAresB0MSE = str(np.round(np.mean((betaMAVec[:, 0] - beta0) ** 2), 3)) + ' (' + str(
        np.round(np.std((betaMAVec[:, 0] - beta0) ** 2), 3)) + ')'
    MAresB1MSE = str(np.round(np.mean((betaMAVec[:, 1] - beta1) ** 2), 5)) + ' (' + str(
        np.round(np.std((betaMAVec[:, 1] - beta1) ** 2), 5)) + ')'

    VLresB0MSE = str(np.round(np.mean((betaVVec[:, 0] - beta0) ** 2), 3)) + ' (' + str(
        np.round(np.std((betaVVec[:, 0] - beta0) ** 2), 3)) + ')'
    VLresB1MSE = str(np.round(np.mean((betaVVec[:, 1] - beta1) ** 2), 5)) + ' (' + str(
        np.round(np.std((betaVVec[:, 1] - beta1) ** 2), 5)) + ')'

    MAresB0MAD = str(np.round(np.mean(np.abs(betaMAVec[:, 0] - beta0)), 3)) + ' (' + str(
        np.round(np.std(np.abs(betaMAVec[:, 0] - beta0)), 3)) + ')'
    MAresB1MAD = str(np.round(np.mean(np.abs(betaMAVec[:, 1] - beta1)), 5)) + ' (' + str(
        np.round(np.std(np.abs(betaMAVec[:, 1] - beta1)), 5)) + ')'

    VLresB0MAD = str(np.round(np.mean(np.abs(betaVVec[:, 0] - beta0)), 3)) + ' (' + str(
        np.round(np.std(np.abs(betaVVec[:, 0] - beta0)), 3)) + ')'
    VLresB1MAD = str(np.round(np.mean(np.abs(betaVVec[:, 1] - beta1)), 5)) + ' (' + str(
        np.round(np.std(np.abs(betaVVec[:, 1] - beta1)), 3)) + ')'

    MAresB0 = str(np.round(np.mean(betaMAVec[:, 0]), 3)) + ' (' + str(np.round(np.std((betaMAVec[:, 0])), 3)) + ')'
    MAresB1 = str(np.round(np.mean(betaMAVec[:, 1]), 3)) + ' (' + str(np.round(np.std((betaMAVec[:, 1])), 3)) + ')'

    VLresB0 = str(np.round(np.mean(betaVVec[:, 0]), 3)) + ' (' + str(np.round(np.std((betaVVec[:, 0])), 3)) + ')'
    VLresB1 = str(np.round(np.mean(betaVVec[:, 1]), 3)) + ' (' + str(np.round(np.std((betaVVec[:, 1])), 3)) + ')'

    return betaVVec, betaMAVec, MAresB0, MAresB1, VLresB0, VLresB1, MAresB0MSE, MAresB1MSE, VLresB0MSE, VLresB1MSE, MAresB0MAD, MAresB1MAD, VLresB0MAD, VLresB1MAD

betaV_25,  betaMA_25,  MAresB0_25,  MAresB1_25,  VLresB0_25,  VLresB1_25,  MAresB0MSE_25,  MAresB1MSE_25,  VLresB0MSE_25,  VLresB1MSE_25,  MAresB0MAD_25,   MAresB1MAD_25,   VLresB0MAD_25,   VLresB1MAD_25   = runSimulation(25)
betaV_50,  betaMA_50,  MAresB0_50,  MAresB1_50,  VLresB0_50,  VLresB1_50,  MAresB0MSE_50,  MAresB1MSE_50,  VLresB0MSE_50,  VLresB1MSE_50,  MAresB0MAD_50,   MAresB1MAD_50,   VLresB0MAD_50,   VLresB1MAD_50   = runSimulation(50)
betaV_100, betaMA_100, MAresB0_100, MAresB1_100, VLresB0_100, VLresB1_100, MAresB0MSE_100, MAresB1MSE_100, VLresB0MSE_100, VLresB1MSE_100, MAresB0MAD_100,  MAresB1MAD_100,  VLresB0MAD_100,  VLresB1MAD_100  = runSimulation(100)
betaV_200, betaMA_200, MAresB0_200, MAresB1_200, VLresB0_200, VLresB1_200, MAresB0MSE_200, MAresB1MSE_200, VLresB0MSE_200, VLresB1MSE_200, MAresB0MAD_200,  MAresB1MAD_200,  VLresB0MAD_200,  VLresB1MAD_200  = runSimulation(200)
betaV_300, betaMA_300, MAresB0_300, MAresB1_300, VLresB0_300, VLresB1_300, MAresB0MSE_300, MAresB1MSE_300, VLresB0MSE_300, VLresB1MSE_300, MAresB0MAD_300,  MAresB1MAD_300,  VLresB0MAD_300,  VLresB1MAD_300  = runSimulation(300)


# Print table column 1
print(VLresB0MAD_25)
print(VLresB0MAD_50)
print(VLresB0MAD_100)
print(VLresB0MAD_200)
print(VLresB0MAD_300)

# Print table column 2
print(VLresB1MAD_25)
print(VLresB1MAD_50)
print(VLresB1MAD_100)
print(VLresB1MAD_200)
print(VLresB1MAD_300)

# Print table column 3
print(MAresB0MAD_25)
print(MAresB0MAD_50)
print(MAresB0MAD_100)
print(MAresB0MAD_200)
print(MAresB0MAD_300)

# Print table column 4
print(MAresB1MAD_25)
print(MAresB1MAD_50)
print(MAresB1MAD_100)
print(MAresB1MAD_200)
print(MAresB1MAD_300)



# Results on MSE scale (Not reported because they get too small at higher sample sizes)
# Print table column 1
print(VLresB0MSE_25)
print(VLresB0MSE_50)
print(VLresB0MSE_100)
print(VLresB0MSE_200)
print(VLresB0MSE_300)

# Print table column 2
print(VLresB1MSE_25)
print(VLresB1MSE_50)
print(VLresB1MSE_100)
print(VLresB1MSE_200)
print(VLresB1MSE_300)

# Print table column 3
print(MAresB0MSE_25)
print(MAresB0MSE_50)
print(MAresB0MSE_100)
print(MAresB0MSE_200)
print(MAresB0MSE_300)

# Print table column 4
print(MAresB1MSE_25)
print(MAresB1MSE_50)
print(MAresB1MSE_100)
print(MAresB1MSE_200)
print(MAresB1MSE_300)








