from extra_sensory_data import ExtraSensoryData
from gmm_ubm import GMM_UBM
import numpy as np
import pandas as pd
import glob
import logging


class Driver(object):
    @staticmethod
    def train_one_model(gmm_train_data, adapt_train_data, model_name, n_comp, max_iter):
        gmm_ubm = GMM_UBM(
            data_source_name=ExtraSensoryData.data_source_name,

            label_name=model_name,
            n_comp=n_comp,
            max_iter=max_iter,
        )
        # train model
        gmm_ubm.fit_ubm(gmm_train_data=gmm_train_data)
        # gmm_ubm.load_ubm()

        # adaptation
        gmm_ubm.fit_adapt(adapt_train_data=adapt_train_data)
        # gmm_ubm.load_adapt()

        return gmm_ubm

    # @staticmethod
    # def load_trained_model():
    #     gmm_ubm = GMM_UBM.load()
    #     return gmm_ubm

    @staticmethod
    def driver_ExtraSensory():
        df_all = ExtraSensoryData.load_df()
        np.random.seed(0)
        msk = np.random.rand(len(df_all)) < 0.8

        df_train = df_all[msk]
        print(ExtraSensoryData.feature_columns_list)

        model = Driver.train_one_model(gmm_train_data=df_train, adapt_train_data=adapt_train_data,
                                       model_name='IN A CAR OR ON A BUS', n_comp=100, max_iter=100)

        raise NotImplementedError("Driver")
        # model= Driver.load_trained_model()

        df_test = df_all[~msk]
        model.predict(df_test)

        # save/display predict result
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    driver = Driver()
    driver.driver_ExtraSensory()
