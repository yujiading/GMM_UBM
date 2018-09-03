import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from extra_sensory_data import ExtraSensoryData
from gmm_ubm import GMM_UBM


class Driver(object):
    DataSourceClass = ExtraSensoryData
    is_load_model_from_disk = True
    is_load_df_from_disk = True

    @staticmethod
    def split_train_valid_test(df_all):
        """
        Split the data based on labels (user_id)
        :param df_all:
        :return: train_x, train_y, valid_x, valid_y, test_x, test_y
        """

        user_ids = df_all[Driver.DataSourceClass.label_column].unique()

        train_user_id, test_user_id = train_test_split(user_ids, test_size=0.2, random_state=1)  # random_state = seed
        train_user_id, valid_user_id = train_test_split(train_user_id, test_size=0.2, random_state=2)

        train_df = df_all[df_all[Driver.DataSourceClass.label_column].isin(train_user_id)]
        valid_df = df_all[df_all[Driver.DataSourceClass.label_column].isin(valid_user_id)]
        test_df = df_all[df_all[Driver.DataSourceClass.label_column].isin(test_user_id)]

        train_x = train_df[Driver.DataSourceClass.feature_columns_list]
        train_y = train_df[Driver.DataSourceClass.label_column]
        valid_x = valid_df[Driver.DataSourceClass.feature_columns_list]
        valid_y = valid_df[Driver.DataSourceClass.label_column]
        test_x = test_df[Driver.DataSourceClass.feature_columns_list]
        test_y = test_df[Driver.DataSourceClass.label_column]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    @staticmethod
    def score_gmm_ubm(gmm_ubm: GMM_UBM, data_x, data_y):
        """
        Adapt gmm_ubm to one user, and check if prediction is True for that user and False for other users
        :param gmm_ubm:
        :param data_x:
        :param data_y:
        :return:
        """
        user_ids = data_y.unique()
        results = []
        labels = []
        for user_id in user_ids:
            results_for_one_adapt = []
            label_for_one_adapt = []
            mask = data_y == user_id
            user_x = data_x[mask]
            user_adapt_x, user_predict_x = train_test_split(user_x, test_size=0.2, random_state=2)

            gmm_ubm.fit_adapt(user_adapt_x)
            predict_result = gmm_ubm.predict_ubm_adapt(test_data=user_adapt_x)
            results_for_one_adapt.append(predict_result)
            label_for_one_adapt.append(True)


            for other_user_id in user_ids:
                if other_user_id == user_id:
                    continue
                mask = data_y == other_user_id
                other_user_x = data_x[mask]
                predict_result = gmm_ubm.predict_ubm_adapt(test_data=other_user_x)
                results_for_one_adapt.append(predict_result)
                label_for_one_adapt.append(False)

            results.extend(results_for_one_adapt)
            labels.extend(label_for_one_adapt)

        y_test = pd.DataFrame(labels)
        y_score = pd.DataFrame(results)

        # calculate the score between result and label
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        average_precision = average_precision_score(y_test, y_score)
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    @staticmethod
    def run():
        df_all = Driver.DataSourceClass.load_df(is_load_df_from_disk=Driver.is_load_df_from_disk)
        train_x, train_y, valid_x, valid_y, test_x, test_y = Driver.split_train_valid_test(df_all)
        gmm_ubm = GMM_UBM(
            data_source_name=Driver.DataSourceClass.__name__,
            model_name='All labels',
            n_comp=100,
            max_iter=100,
            balance_factor=0.5
        )
        if Driver.is_load_model_from_disk:
            gmm_ubm.load_background_gmm()
        else:
            gmm_ubm.fit_background_gmm(train_x=train_x)

        Driver.score_gmm_ubm(gmm_ubm=gmm_ubm, data_x=valid_x, data_y=valid_y)

        raise NotImplementedError("Driver")

        # save/display predict result
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    driver = Driver()
    driver.run()
