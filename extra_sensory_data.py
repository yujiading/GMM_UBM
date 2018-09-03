import numpy as np
import pandas as pd
import glob


class ExtraSensoryData(object):
    data_source_name= 'extra_sensory_'
    is_load_raw_df_from_disk = True
    is_load_processed_df_from_disk = True

    label_columns_list = ['label:IN_A_CAR', 'label:ON_A_BUS']

    raw_feature_columns_dict = {
        'acc': ['raw_acc:3d:mean_x', 'raw_acc:3d:mean_y', 'raw_acc:3d:mean_z'],
        'gyro': ['proc_gyro:3d:mean_x', 'proc_gyro:3d:mean_y', 'proc_gyro:3d:mean_z']
    }
    # One GMM_UBM per feature
    feature_columns_dict = {
        'normal_acc': ['normal_acc_x', 'normal_acc_y', 'normal_acc_z'],
        'normal_gyro': ['normal_gyro_x', 'normal_gyro_y', 'normal_gyro_z'],
        'magni_acc': ['magni_acc'],
        'magni_gyro': ['magni_gyro'],
    }
    feature_columns_list = [item for sublist in feature_columns_dict.values() for item in sublist]
    feature_training_configs = {
        'normal_acc': {
            'n_component': 100,
            'n_iter': 100
        },
        'normal_gyro': {
            'n_component': 100,
            'n_iter': 100
        },
        'magni_acc': {
            'n_component': 100,
            'n_iter': 100
        },
        'magni_gyro': {
            'n_component': 100,
            'n_iter': 100
        }
    }

    @staticmethod
    def _load_raw_data(raw_columns):
        raw_data_file_name = ExtraSensoryData.data_source_name + "raw_df.pkl"

        if ExtraSensoryData.is_load_raw_df_from_disk:
            df_raw = pd.read_pickle(raw_data_file_name)
            return df_raw

        # data extracting
        user_id = 0
        df_list = []
        path = r"datasets/ExtraSensory"
        allfiles = glob.glob(path + "/*.csv")
        for filename in allfiles:
            df_raw_one_file = pd.read_csv(filename, header=0, usecols=raw_columns)

            df_raw_one_file['users'] = user_id + 1
            df_raw_one_file = df_raw_one_file.dropna()
            # dff=dff[dff['label:ON_A_BUS'].notnull()]
            # dff=dff[dff['label:IN_A_CAR'].notnull()]
            df_list.append(df_raw_one_file)
            user_id += 1

        df_raw = pd.concat(df_list, ignore_index=True)  # type: pd.DataFrame
        df_raw.to_pickle(raw_data_file_name)
        return df_raw

    @staticmethod
    def _raw_column_to_feature_column_name(raw_column_name):
        feature_column_name = raw_column_name.replace("raw_", "normal_")
        feature_column_name = feature_column_name.replace("proc_", "normal_")
        feature_column_name = feature_column_name.replace(":3d:mean_", "_")
        return feature_column_name

    @staticmethod
    def _process_features(df_raw: pd.DataFrame):
        processed_df_file_name = ExtraSensoryData.data_source_name + "processed_df.pkl"
        if ExtraSensoryData.is_load_processed_df_from_disk:
            df_full = pd.read_pickle(processed_df_file_name)
            return df_full

        df_processed = df_raw
        # normalize features
        for feature_name, feature_columns in ExtraSensoryData.raw_feature_columns_dict.items():
            magnitued_name = 'magni_' + feature_name

            magni = np.sqrt(np.square(df_processed[feature_columns]).sum(axis=1))
            df_processed[magnitued_name] = magni

            for feature_column in feature_columns:
                feature_column_name = ExtraSensoryData._raw_column_to_feature_column_name(
                    raw_column_name=feature_column)
                df_processed[feature_column_name] = df_processed[feature_column] / df_processed[magnitued_name]

        # extract processed feature columns
        df_columns = ExtraSensoryData.feature_columns_list + ExtraSensoryData.label_columns_list
        df_processed = df_processed[df_columns]
        print(df_processed.info())
        df_processed.to_pickle(processed_df_file_name)

        return df_processed

    @staticmethod
    def load_df():
        nested_raw_columns = ExtraSensoryData.raw_feature_columns_dict.values()
        raw_feature_columns = [item for sublist in nested_raw_columns for item in sublist]
        raw_columns = raw_feature_columns + ExtraSensoryData.label_columns_list

        print(raw_columns)
        df_raw = ExtraSensoryData._load_raw_data(raw_columns=raw_columns)
        df_all = ExtraSensoryData._process_features(df_raw=df_raw)
        return df_all
