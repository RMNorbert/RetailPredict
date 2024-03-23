import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from data_processor.data_preparator import sales_dates, monthly_sales, pd
from data_processor.model_data_preprocessor import x_train, y_train, x_test
from model.model_helper import calculate_rmse, calculate_mae, calculate_r2, create_predict_series, create_predict_test_set

MODEL_NAME = 'Random Forest'

config_files = ['data_processor/data_configuration.yaml',
                'model/random_forest/random_forest_configuration.yaml']

config_data = {}
for file_path in config_files:
    with open(file_path, 'r') as file:
        for key, value in yaml.safe_load(file).items():
            config_data[key] = value

# Instantiate Random Forest regressor
rf_model = RandomForestRegressor()

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf_model,
                           param_grid=config_data['PARAM_GRID'],
                           cv=config_data['CV_SPLITTING_STRATEGY'],
                           n_jobs=config_data['PARALLEL_JOBS'],
                           verbose=config_data['VERBOSITY'])

# Fit the GridSearchCV to the data
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
rf_model_best = RandomForestRegressor(**best_params)
rf_model_best.fit(x_train, y_train)

# Predict sales
rf_predict = rf_model_best.predict(x_test)

# Scale restore
rf_predict = rf_predict.reshape(
    config_data['RESHAPE_ROW_NUMBER'], config_data['RESHAPE_COLUMN_NUMBER'])

rf_predict_test_set = create_predict_test_set(
    rf_predict, x_test, config_data['AXIS_COLUMN'])

# Append predicted sale values
rf_predict_series = create_predict_series(
    rf_predict_test_set, config_data['RANDOM_FOREST_PREDICT_COLUMN'])

predict_df = pd.DataFrame({config_data['DATE_COLUMN']: sales_dates,
                          config_data['RANDOM_FOREST_PREDICT_COLUMN']: rf_predict_series})

# Evaluation metrics calculation
rf_rmse = calculate_rmse(predict_df, config_data['RANDOM_FOREST_PREDICT_COLUMN'],
                         monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])

rf_mae = calculate_mae(predict_df, config_data['RANDOM_FOREST_PREDICT_COLUMN'],
                       monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])

rf_r2 = calculate_r2(predict_df, config_data['RANDOM_FOREST_PREDICT_COLUMN'],
                     monthly_sales, config_data['SALES_COLUMN'], config_data['PREDICTION_PERIOD'])
