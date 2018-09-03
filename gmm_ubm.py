import os
import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.externals import joblib
import glob
import logging
import scipy.stats
from sklearn.base import clone


# for feature_name, feature_training_config in ExtraSensoryData.feature_training_configs.items():

class GMM_UBM(object):
    """
    Train one GMM for UBM model. Adapt one GMM from UBM for client model.
    return one client adapted gmm model.
    """

    def __init__(self,
                 data_source_name: str,
                 label_name: str,
                 n_comp,
                 max_iter,
                 score_thresh,
                 balance_factor,
                 ):
        self.logger = logging.getLogger(__name__)  # __name__ is file name; GMM_UBM.__name__ is class name
        self.model_path_prefix = os.path.join(data_source_name + "models", label_name)
        self.n_comp = n_comp
        self.max_iter = max_iter
        self.score_thresh = score_thresh
        self.balance_factor=balance_factor
        self.gmm: mixture.GaussianMixture = None
        self.adapt_model: mixture.GaussianMixture = None
        self.score_lld = []

    def fit_ubm(self, gmm_train_data: pd.DataFrame):
        gmm = mixture.GaussianMixture(n_components=self.n_comp,
                                      covariance_type='diag',
                                      max_iter=self.max_iter,
                                      verbose=2, reg_covar=1e-9)
        self.logger.info("Training data with columns: {feature_columns}".format(
            feature_columns=list(gmm_train_data)))
        gmm.fit(gmm_train_data)

        if not os.path.exists(self.model_path_prefix):
            os.makedirs(self.model_path_prefix)
        joblib.dump(gmm, os.path.join(self.model_path_prefix, 'gmm.pkl'))
        # self.extract_param()
        self.gmm = gmm

    def load_ubm(self):
        gmm = joblib.load(os.path.join(self.model_path_prefix, 'gmm.pkl'))
        self.gmm = gmm

    def _one_pie_n(self, model, gmm_class, obs):
        one_pie = model.weights_[gmm_class]
        one_n = scipy.stats.multivariate_normal(model.means_[gmm_class],
                                                np.diag(model.covariances_[gmm_class])).pdf(obs)
        return one_pie * one_n

    def _obs_to_pie_n_row(self, obs):
        one_row = []
        for gmm_class in range(self.n_comp):
            one_row.append(self._one_pie_n(self.gmm, gmm_class, obs))
        return one_row

    def _npz_to_e_one(self, one_n, col_p, adapt_train_data):
        one_e = adapt_train_data.mul(col_p, axis=0).sum() / one_n
        return one_e

    def fit_adapt(self, adapt_train_data: pd.DataFrame):
        self.adapt_model = clone(self.gmm)
        pie_n = adapt_train_data.apply(self._obs_to_pie_n_row, result_type='expand', axis=1)  # type: pd.DataFrame
        p = pie_n.div(pie_n.sum(axis=1), axis='rows')
        n = p.sum()
        e_list = [self._npz_to_e_one(n[gmm_class], p.iloc[:, gmm_class], adapt_train_data) for gmm_class in
                  range(self.n_comp)]
        e = pd.concat(e_list, axis=1)
        alpha = n / (n + self.balance_factor)
        for gmm_class in range(self.n_comp):
            if alpha[gmm_class] == 0:
                new_mean = self.adapt_model.means_[gmm_class]
            else:
                new_mean = alpha[gmm_class] * e.iloc[:, gmm_class] + (1 - alpha[gmm_class]) * self.adapt_model.means_[
                    gmm_class]
            self.adapt_model.means_[gmm_class] = new_mean

        if not os.path.exists(self.model_path_prefix):
            os.makedirs(self.model_path_prefix)
        joblib.dump(self.adapt_model, os.path.join(self.model_path_prefix, 'adapt.pkl'))
        # self.extract_param()

    def load_adapt(self):
        adapt_model = joblib.load(os.path.join(self.model_path_prefix, 'adapt.pkl'))
        self.adapt_model = adapt_model

    def score_ubm_adapt(self, test_data: pd.DataFrame):
        score = self.adapt_model.score(test_data.values[:], y=None) - self.gmm.score(test_data.values[:], y=None)
        self.score_lld = score

    def predict_ubm_adapt(self):
        if self.score_lld >= self.score_thresh:
            return True


a = 0
# def load_model(self):
#     self.gmm = joblib.load('models/ubm.pkl')
#     self.extract_param()
#
# def extract_param(self):
#     self.means = self.gmm.means_
#     self.covars = self.gmm.covariances_
#     self.weights = self.gmm.weights_
#     self.logl = self.gmm.lower_bound_
#
# def dump_paras(self, dirname):
#     with open(dirname + "/ubm_means", 'w') as f:
#         for vec in self.means:
#             f.write(' '.join(map(str, vec)))
#             f.write('\n')
#         f.write('\n')
#     with open(dirname + "/ubm_variances", 'w') as f:
#         for mat in self.covars:
#             for d in range(self.n_comp):
#                 f.write(str(mat[d][d]))
#                 f.write(' ')
#             f.write('\n')
#         f.write('\n')
#     with open(dirname + "/ubm_weights", 'w') as f:
#         f.write(' '.join(map(str, self.weights)))
#         f.write('\n')
#
# def simulate_data(self):
#     n_samples = 300
#
#     # generate random sample, two components
#     np.random.seed(0)
#
#     # generate spherical data centered on (20, 20)
#     shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
#
#     # generate zero centered stretched Gaussian data
#     C = np.array([[0., -0.7], [3.5, .7]])
#     stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
#
#     # concatenate the two datasets into the final training set
#     X_train = np.vstack([shifted_gaussian, stretched_gaussian])
#     self.df_train = X_train


# if __name__ == "__main__":
#     ubm = UBM(2, 100)
#     ubm.simulate_data()
#     ubm.fit()
#     # ubm = joblib.load('ubm.pkl')
#     model_dir = 'models'
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     joblib.dump(ubm, 'models/ubm.pkl')
#     param_dir = 'parameters'
#     if not os.path.exists(param_dir):
#         os.makedirs(param_dir)
#     ubm.dump_paras(param_dir)
