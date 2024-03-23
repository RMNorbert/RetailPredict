import yaml
from sklearn.linear_model import LinearRegression
from data_processor.data_preparator import sales_dates, pd, monthly_sales
from data_processor.model_data_preprocessor import x_train, y_train, x_test
from model.model_helper import calculate_rmse, calculate_mae, calculate_r2, create_predict_series, create_predict_test_set

with open('data_processor/data_configuration.yaml', 'r') as file:
    data = yaml.safe_load(file)

MODEL_NAME = 'Linear Regression'

# Train and predict
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_predict = lr_model.predict(x_test)

# Scale restore
lr_predict = lr_predict.reshape(
    data['RESHAPE_ROW_NUMBER'], data['RESHAPE_COLUMN_NUMBER'])

lr_predict_test_set = create_predict_test_set(
    lr_predict, x_test, data['AXIS_COLUMN'])

# Append predicted sale values
lr_predict_series = create_predict_series(
    lr_predict_test_set, data['LINEAR_PREDICT_COLUMN'])

predict_df = pd.DataFrame(
    {data['DATE_COLUMN']: sales_dates, data['LINEAR_PREDICT_COLUMN']: lr_predict_series})

# Evaluation metrics calculation
lr_rmse = calculate_rmse(predict_df, data['LINEAR_PREDICT_COLUMN'],
                         monthly_sales, data['SALES_COLUMN'], data['PREDICTION_PERIOD'])

lr_mae = calculate_mae(predict_df, data['LINEAR_PREDICT_COLUMN'],
                       monthly_sales, data['SALES_COLUMN'], data['PREDICTION_PERIOD'])

lr_r2 = calculate_r2(predict_df, data['LINEAR_PREDICT_COLUMN'],
                     monthly_sales, data['SALES_COLUMN'], data['PREDICTION_PERIOD'])
