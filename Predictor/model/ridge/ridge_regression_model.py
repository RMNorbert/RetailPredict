import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from data_processor.data_preparator import sales_dates, monthly_sales, pd
from data_processor.model_data_preprocessor import x_train, y_train, x_test
from model.model_helper import calculate_rmse, calculate_mae, calculate_r2, create_predict_test_set, create_predict_series

MODEL_NAME = 'Ridge Regression'

config_files = ['data_processor/data_configuration.yaml',
                'model/ridge/ridge_configuration.yaml']

config_data = {}
for file_path in config_files:
    with open(file_path, 'r') as file:
        for key, value in yaml.safe_load(file).items():
            config_data[key] = value

# Instantiate Ridge regressor
ridge_model = Ridge()

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=ridge_model,
                           param_grid=config_data['PARAM_GRID'],
                           cv=config_data['CV_SPLITTING_STRATEGY'],
                           n_jobs=config_data['PARALLEL_JOBS'],
                           verbose=config_data['VERBOSITY'])

# Fit the GridSearchCV to the data
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
ridge_model_best = Ridge(**best_params)
ridge_model_best.fit(x_train, y_train)

# Predict sales
ridge_predict = ridge_model_best.predict(x_test)

# Scale restore
ridge_predict = ridge_predict.reshape(config_data['RESHAPE_ROW_NUMBER'], config_data['RESHAPE_COLUMN_NUMBER'])
ridge_predict_test_set = create_predict_test_set(ridge_predict, x_test, config_data['AXIS_COLUMN'])

# Append predicted sale values
ridge_predict_series = create_predict_series(ridge_predict_test_set, config_data['LINEAR_PREDICT_COLUMN'])
predict_df = pd.DataFrame({config_data['DATE_COLUMN']: sales_dates, config_data['LINEAR_PREDICT_COLUMN']: ridge_predict_series})

# Evaluation metrics calculation
ridge_rmse = calculate_rmse(predict_df, config_data['LINEAR_PREDICT_COLUMN'], monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])
ridge_mae = calculate_mae(predict_df, config_data['LINEAR_PREDICT_COLUMN'], monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])
ridge_r2 = calculate_r2(predict_df, config_data['LINEAR_PREDICT_COLUMN'], monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])
