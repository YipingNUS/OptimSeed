import numpy as np


class BEE:
    """ Bayesian error estimation.
        Try to implement the Bayesian Error Estimation (BEE) model in following paper.
        Note:
            1) this method only works for binary classification and
            2) assumes the majority of the classifiers have an error rate < 0.5

        I realized the author provided official implementation. It's in Java and
        not very trivial to read: https://github.com/eaplatanios/makina

        Reference:
            Emmanouil Antonios Platanios, Avinava Dubey, Tom Mitchell ; Proceedings of The 33rd International Conference on Machine Learning, PMLR 48:1416-1425, 2016.
    """
    max_p, min_p = 0.85, 0.15  # parameter to filter out non-sense estimators predicting extreme values

    # TODO: set the proper hyper-parameter
    # Values from the paper, assuming the error rates are generally small
    alpha_p, beta_p, alpha_e, beta_e = 1.0, 1.0, 1.0, 10.0

    def __init__(self, labeling_matrix, num_iters=50, init_method='sampling', filter_estimators=False, filter_by_std=False):
        """ Initialize the BEE model

        :param labeling_matrix: the prediction matrix [num_samples, num_estimators]. Each entry is either 0 or 1
        :param num_iters: the number of Gibbs sampling iterations
        :param init_method: [maj, sampling]
        """

        if filter_estimators:
            labeling_matrix, estimator_indices, num_orig_estimators = self.filter_estimators(labeling_matrix, filter_by_std=filter_by_std)
        self.labeling_matrix = labeling_matrix
        self.num_samples = labeling_matrix.shape[0]
        self.num_estimators = labeling_matrix.shape[1]
        self.num_iters = num_iters
        self.true_labels = np.zeros(self.num_samples)
        self.error_rates = np.zeros(self.num_estimators)

        if init_method == 'maj':  # initialize the true label with the majority voting
            self.initialize_label_majority_vote(labeling_matrix)

        elif init_method == 'sampling':  # sample the initial true label from the prior
            self.initialize_label_gibbs_sampling()

        self.gibbs_sampling()
        if filter_estimators:
            self.fix_error_rate(estimator_indices, num_orig_estimators)

    def initialize_label_majority_vote(self, labeling_matrix):
        """ initialize the initial true label by majority voting

        :param labeling_matrix: the prediction matrix [num_samples, num_estimators]. Each entry is either 0 or 1
        """
        row_avg = np.mean(labeling_matrix, axis=1)
        self.true_labels = (row_avg > 0.5).astype(int)
        self.error_rates = self.__calc_err_rate()

    def initialize_label_gibbs_sampling(self):
        """ iniialize the initial true label by sampling from prior
        """
        for i in range(self.num_samples):
            p = np.random.beta(BEE.alpha_p, BEE.beta_p)
            self.true_labels[i] = np.random.binomial(1, p)
        for j in range(self.num_estimators):
            self.error_rates[j] = np.random.beta(BEE.alpha_e, BEE.beta_e)

    def __calc_err_rate(self):
        """ calculate the current error rate of estimators"""
        return np.mean(self.labeling_matrix == self.true_labels.reshape(-1, 1), axis=0)

    def sample_p(self):
        """ equation 2
        """
        sigma_l = np.sum(self.true_labels)
        return np.random.beta(BEE.alpha_p + sigma_l, BEE.beta_p + self.num_samples - sigma_l)

    def sample_p_discount(self, i):
        """ equation 2 + discounting the old label when sampling
        """
        sigma_l = np.sum(self.true_labels) - self.true_labels[i]
        return np.random.beta(BEE.alpha_p + sigma_l, BEE.beta_p + self.num_samples - sigma_l - 1)

    def sample_l(self, p, i):
        """ equation 3
        """
        # the number of correct predictions of each estimator. dim [num_estimators, 1]
        pi = np.zeros(2)  # the pi value for l=0 and l=1
        for k in range(2):
            num_corrects = self.labeling_matrix[i, :] == k
            temp = np.power(self.error_rates, 1 - num_corrects) * np.power(1 - self.error_rates, num_corrects)
            pi[k] = np.prod(temp)

        prob = pi * np.asarray([1 - p, p])
        positive_prob = prob[1] / np.sum(prob)

        return np.random.binomial(1, positive_prob)

    def sample_e(self, j):
        """ equation 4
        """
        sigma_j = np.sum(self.labeling_matrix[:, j] != self.true_labels)
        return np.random.beta(BEE.alpha_e + sigma_j, BEE.beta_e + self.num_samples - sigma_j)

    def gibbs_sampling(self):
        """ perform the main Gibbs sampling """
        for it in range(self.num_iters):
            for i in range(self.num_samples):
                p = self.sample_p()
                #p = self.sample_p_discount(i)
                self.true_labels[i] = self.sample_l(p, i)
            for j in range(self.num_estimators):
                self.error_rates[j] = self.sample_e(j)

    def filter_estimators(self, labeling_matrix, filter_by_std=False):
        """ filter estimators. remove outliers to make BEE more robust.
        This method will remove columns from self.labeling_matrix representing poor estimators

        :return: indices of the selected estimator, orig_num_estimators
        """
        empirical_p = np.mean(labeling_matrix, axis=0)
        empirical_p = np.where(((empirical_p > BEE.min_p) & (empirical_p < BEE.max_p)), empirical_p, -1)
        if filter_by_std:
            mean, std = np.mean(empirical_p[empirical_p>0]), np.std(empirical_p[empirical_p>0])
            empirical_p = np.where(((empirical_p > mean - std) & (empirical_p < mean + std)), empirical_p, -1)
        selected_indices = np.argwhere(empirical_p>0).reshape(-1,)
        orig_num_estimators = labeling_matrix.shape[1]
        labeling_matrix = labeling_matrix[:, selected_indices]
        return labeling_matrix, selected_indices, orig_num_estimators

    def fix_error_rate(self, estimator_indices, num_orig_estimators):
        """ fix the error rate. Set error rate to 1 for neglected estimators
        :arg estimator_indices - the indices of selected estimators
        :arg num_orig_estimators - number of original estimators
        :return:
        """
        assert self.error_rates.shape == estimator_indices.shape
        num_actual_estimators = estimator_indices.shape[0]
        if num_actual_estimators == num_orig_estimators:
            return
        temp_error_rate = np.ones(num_orig_estimators)
        for i in range(num_actual_estimators):
            temp_error_rate[estimator_indices[i]] = self.error_rates[i]
        #print("[INFO] Estimated error rate before fixing:", self.error_rates)
        self.error_rates = temp_error_rate
        #print("[INFO] Estimated error rate after fixing:", self.error_rates)
