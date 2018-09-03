import os
import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.externals import joblib
import glob
import logging


# for feature_name, feature_training_config in ExtraSensoryData.feature_training_configs.items():

class GMM_UBM(object):
    """
    Train one GMM per feature (a set of columns).
    Add the probability for prediction from each GMM for final prediction score.
    """

    def __init__(self,
                 data_source_name: str,
                 train_data: pd.DataFrame,
                 label_name: str,
                 feature_training_configs: dict,
                 feature_columns_dict: dict):
        self.logger = logging.getLogger(__name__)
        self.df_train = train_data
        self.model_path_prefix = os.path.join(data_source_name + "models", label_name)
        self.feature_training_configs = feature_training_configs
        self.feature_columns_dict = feature_columns_dict
        self.gmm_dict_by_feature = {}

    def fit_gmms(self):
        """

        :return:
        """
        for feature_name, feature_columns in self.feature_columns_dict.items():
            training_config = self.feature_training_configs[feature_name]
            gmm = self._fit_gmm(feature_name=feature_name,
                                training_config=training_config,
                                feature_columns=feature_columns)
            self.gmm_dict_by_feature[feature_name] = gmm

    def _fit_gmm(self, feature_name, training_config, feature_columns):
        n_comp = training_config['n_component']  # training_config is sublist of feature_training_configs
        n_iter = training_config['n_iter']
        gmm = mixture.GaussianMixture(n_components=n_comp,
                                      covariance_type='diag',
                                      max_iter=n_iter,
                                      verbose=2, reg_covar=1e-9)
        gmm_training_data = self.df_train[feature_columns]
        self.logger.info("Training feature {feature_name} with columns: {feature_columns}".format(
            feature_name=feature_name, feature_columns=feature_columns))
        gmm.fit(gmm_training_data)

        if not os.path.exists(self.model_path_prefix):
            os.makedirs(self.model_path_prefix)
        model_file_name = feature_name + '.pkl'
        joblib.dump(gmm, os.path.join(self.model_path_prefix, model_file_name))
        # self.extract_param()
        return gmm

    def load_gmms(self):
        for feature_name in self.feature_columns_dict:
            model_file_name = feature_name + '.pkl'
            gmm = joblib.load(model_file_name)
            self.gmm_dict_by_feature[feature_name] = gmm

    def adaptation(self):

        pass

    def predict_gmms(self):
        pass

    def _predict_gmm(self):
        pass


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
