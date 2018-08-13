from extra_sensory_data import ExtraSensoryData
from gmm_ubm import GMM_UBM
import numpy as np
import pandas as pd
import glob
import logging


class Driver(object):
    @staticmethod
    def train_model(df_train, model_name):
        gmm_ubm = GMM_UBM(
            data_source_name=ExtraSensoryData.data_source_name,
            train_data=df_train,
            label_name=model_name,
            feature_columns_dict=ExtraSensoryData.feature_columns_dict,
            feature_training_configs=ExtraSensoryData.feature_training_configs,
        )
        # train model
        gmm_ubm.fit_gmms()
        # gmm_ubm.load_gmms()

        # adaptation
        gmm_ubm.adaptation()

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

        models_dict_by_label = {}
        for label_column in ExtraSensoryData.label_columns_list:
            df_train_model = df_train[df_train[label_column] == 1]
            model = Driver.train_model(df_train=df_train_model, model_name=label_column)
            models_dict_by_label[label_column] = model

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
