from runMethods import run_ma, run_aug, run_vg, run_vl, run_gm, run_mb


class ModelObject:
    def __init__(self, data, policy, policy_gen, trans_model, feature_model, dr_params, vl_params,
                 reference_distribution, discount, max_procs):
        self.data = data
        self.policy = policy
        self.policy_gen = policy_gen
        self.trans_model = trans_model
        self.feature_model = feature_model
        self.dr_params = dr_params
        self.vl_params = vl_params
        self.reference_distribution = reference_distribution
        self.discount = discount
        self.max_procs = max_procs

    def set_policy(self, policy):
        self.policy = policy

    def run_v_learning(self, policy=None):
        if policy is None:
            policy = self.policy
        value = run_vl(self.data, policy, self.feature_model, self.vl_params, self.discount,
                       self.reference_distribution)
        return value

    def run_godambe_modeled(self, policy=None):
        if policy is None:
            policy = self.policy
        value = run_gm(self.data, policy, self.feature_model, self.discount, self.reference_distribution)
        return value

    def run_model_based(self, policy=None):
        if policy is None:
            policy = self.policy
        value = run_mb(policy, self.trans_model, self.dr_params, self.discount, self.reference_distribution,
                       self.max_procs)
        return value

    def run_model_assisted(self, policy=None):
        if policy is None:
            policy = self.policy
        value = run_ma(self.data, self.policy_gen, policy, self.feature_model, self.trans_model, self.vl_params,
                       self.dr_params, self.discount, self.reference_distribution, self.max_procs)
        return value

    def run_augmented(self, policy=None):
        if policy is None:
            policy = self.policy
        value = run_aug(self.data, self.policy_gen, policy, self.trans_model, self.feature_model, self.vl_params,
                        self.reference_distribution, self.discount)
        return value

    def run_godambe_mc(self, policy=None):
        if policy is None:
            policy = self.policy
        value = run_vg(self.data, self.policy_gen, policy, self.feature_model, self.trans_model, self.vl_params,
                       self.dr_params, self.discount, self.reference_distribution, self.max_procs)
        return value

    def fit_model(self, method='VL', policy=None):
        if policy is None:
            policy = self.policy
        if method == 'VL':
            value = self.run_v_learning(policy)
        elif method == 'MB':
            value = self.run_model_based(policy)
        elif method == 'AG':
            value = self.run_augmented(policy)
        elif method == 'GM':
            value = self.run_godambe_modeled(policy)
        elif method == 'VG':
            value = self.run_godambe_mc(policy)
        elif method == 'MA':
            value = self.run_model_assisted(policy)
        else:
            value = 0
        return value
