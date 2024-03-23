import yaml
import xgboost as xgb
from data_processor.data_preparator import sales_dates, monthly_sales, pd
from data_processor.model_data_preprocessor import x_train, y_train, x_test, y_test
from model.model_helper import calculate_rmse, calculate_mae, calculate_r2, create_predict_series, create_predict_test_set

MODEL_NAME = 'XG Boost'

config_files = ['data_processor/data_configuration.yaml',
                'model/xg_boost/xg_boost_configuration.yaml']

config_data = {}
for file_path in config_files:
    with open(file_path, 'r') as file:
        for key, value in yaml.safe_load(file).items():
            config_data[key] = value


# Prepare data
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# Train the model
model = xgb.train(
    config_data['PARAMS'],
    dtrain,
    num_boost_round=config_data['BOOST_ROUND_NUMBER'],
    evals=[(dtest, config_data['TEST_EVALUATION_SET_NAME'])],
    early_stopping_rounds=config_data['STOPPING_ROUNDS']
)

# Predict sales
xgb_predict = model.predict(dtest)

# Scale restore
xgb_predict = xgb_predict.reshape(
    config_data['RESHAPE_ROW_NUMBER'], config_data['RESHAPE_COLUMN_NUMBER'])

xgb_predict_test_set = create_predict_test_set(
    xgb_predict, x_test, config_data['AXIS_COLUMN'])

# Append predicted sale values
xgb_predict_series = create_predict_series(
    xgb_predict_test_set, config_data['XGB_PREDICT_COLUMN'])

# Create DataFrame with predictions
predict_df = pd.DataFrame(
    {config_data['DATE_COLUMN']: sales_dates, config_data['XGB_PREDICT_COLUMN']: xgb_predict_series})

# Evaluation metrics
xgb_rmse = calculate_rmse(predict_df, config_data['XGB_PREDICT_COLUMN'],
                          monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])

xgb_mae = calculate_mae(predict_df, config_data['XGB_PREDICT_COLUMN'],
                        monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])

xgb_r2 = calculate_r2(predict_df, config_data['XGB_PREDICT_COLUMN'], monthly_sales,
                      config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])
