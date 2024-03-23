from data_processor.data_preparator import monthly_sales
from service.visualizer.pyplot.pyplot_data_visualizer import visualize_sales
from model.linear.linear_regression_model import predict_df as lr_df
from model.random_forest.random_forest_model import predict_df as rf_df
from model.xg_boost.xgboost_model import predict_df as xgb_df
from model.ridge.ridge_regression_model import predict_df as rr_df
import sys


def visualize(model_type):
    match model_type:
        case 'linear':
            visualize_sales(monthly_sales, lr_df)
        case 'ridge':
            visualize_sales(monthly_sales, rr_df)
        case 'forest':
            visualize_sales(monthly_sales, rf_df)
        case 'xgb':
            visualize_sales(monthly_sales, xgb_df)
        case _:
            print(f'Unsupported model provided {model_type}')


if __name__ == "__main__":
    model = 'linear'
    if len(sys.argv) == 2:
        model = sys.argv[1]
        visualize(model)

    visualize(model)
