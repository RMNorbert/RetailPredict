import yaml
import matplotlib.pyplot as plt

from custom_error.column_not_found_error import ColumnNotFoundError

config_files = ['Predictor/data_processor/data_configuration.yaml',
                'Predictor/service/visualizer/pyplot/pyplot_configuration.yaml']

config_data = {}
for file_path in config_files:
    with open(file_path, 'r') as file:
        for key, value in yaml.safe_load(file).items():
            config_data[key] = value


def get_column_for_visualization(df):
    columns_to_try = [config_data['LINEAR_PREDICT_COLUMN'],
                      config_data['RIDGE_PREDICT_COLUMN'],
                      config_data['RANDOM_FOREST_PREDICT_COLUMN'],
                      config_data['XGB_PREDICT_COLUMN']]
    column_to_plot = None

    for column in columns_to_try:
        try:
            column_to_plot = df[column]
            break
        except KeyError:
            pass

    if column_to_plot is None:
        raise ColumnNotFoundError()
    else:
        return column_to_plot


def get_title(column_to_plot):
    if column_to_plot.name == config_data['LINEAR_PREDICT_COLUMN']:
        return config_data['LINEAR_REGRESSION_SUBPLOT_TITLE']
    elif column_to_plot.name == config_data['RIDGE_PREDICT_COLUMN']:
        return config_data['RIDGE_REGRESSION_SUBPLOT_TITLE']
    elif column_to_plot.name == config_data['RANDOM_FOREST_PREDICT_COLUMN']:
        return config_data['RANDOM_FOREST_SUBPLOT_TITLE']
    else:
        return config_data['XGB_SUBPLOT_TITLE']


# visualizing the monthly sales with predicted sales and sales difference
def visualize_sales(monthly_sales, predict_df):
    column_to_plot = get_column_for_visualization(predict_df)
    title_to_set = get_title(column_to_plot)

    fig, (sales, sales_diff) = plt.subplots(config_data['ROWS'],
                                            config_data['COLUMNS'],
                                            figsize=(config_data['WIDTH'], config_data['HEIGHT'])
                                            )
    sales.plot(monthly_sales[config_data['DATE_COLUMN']], monthly_sales[config_data['SALES_COLUMN']])
    sales.plot(predict_df[config_data['DATE_COLUMN']], column_to_plot)
    sales.set_xlabel(config_data['X_LABEL'])
    sales.set_ylabel(config_data['Y_LABEL'])
    sales.set_title(title_to_set)
    sales.legend([config_data['ORIGINAL_LEGEND'], config_data['PREDICTION_LEGEND']])
    sales_diff.plot(monthly_sales[config_data['DATE_COLUMN']], monthly_sales[config_data['SALES_DIFFERENCE_COLUMN']])
    sales_diff.set_xlabel(config_data['X_LABEL'])
    sales_diff.set_ylabel(config_data['Y_LABEL'])
    sales_diff.set_title(config_data['SECOND_SUBPLOT_TITLE'])
    plt.show()


# visualizing different model performance
def visualize_model_comparison(model1_name, model1_stats,
                               model2_name, model2_stats,
                               model3_name, model3_stats,
                               model4_name, model4_stats):
    plt.figure(figsize=(config_data['WIDTH'], config_data['HEIGHT']))
    plt.plot(model1_stats)
    plt.plot(model2_stats)
    plt.plot(model3_stats)
    plt.plot(model4_stats)
    plt.title(config_data['MODEL_COMPARE_SUBPLOT_TITLE'])
    plt.xticks(config_data['X_AXIS_COMPARISON_LABELS_POSITIONS'], labels=config_data['MODEL_COMPARISON_LABELS'])
    plt.legend([model1_name, model2_name, model3_name, model4_name])
    plt.show()
