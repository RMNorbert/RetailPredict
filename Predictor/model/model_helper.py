import numpy as np
from data_processor.data_preparator import act_sales, pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_processor.model_data_preprocessor import scaler


def create_predict_test_set(rf_predict, x_test, axis):
    predict_test_set = np.concatenate([rf_predict, x_test], axis=axis)
    predict_test_set = scaler.inverse_transform(predict_test_set)
    return predict_test_set


def create_predict_list(predict_test_set):
    predict_list = []
    for index in range(0, len(predict_test_set)):
        predict_list.append(predict_test_set[index][0] + act_sales[index])
    return predict_list


def create_predict_series(predict_test_set, column_name):
    result_list = create_predict_list(predict_test_set)
    return pd.Series(result_list, name=column_name)


def calculate_rmse(predict_df, predict_column_name, actual_dataset, actual_column_name, period):
    return np.sqrt(mean_squared_error(predict_df[predict_column_name], actual_dataset[actual_column_name][period:]))


def calculate_mae(predict_df, predict_column_name, actual_dataset, actual_column_name, period):
    return mean_absolute_error(predict_df[predict_column_name],  actual_dataset[actual_column_name][period:])


def calculate_r2(predict_df, predict_column_name, actual_dataset, actual_column_name, period):
    return r2_score(predict_df[predict_column_name], actual_dataset[actual_column_name][period:])
