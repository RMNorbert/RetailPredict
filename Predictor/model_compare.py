from model.linear.linear_regression_model import MODEL_NAME as LR_NAME, lr_rmse, lr_mae, lr_r2
from model.ridge.ridge_regression_model import MODEL_NAME as RR_NAME, ridge_rmse, ridge_mae, ridge_r2
from service.visualizer.pyplot.pyplot_data_visualizer import visualize_model_comparison
from model.random_forest.random_forest_model import MODEL_NAME as RF_NAME, rf_rmse, rf_mae, rf_r2
from model.xg_boost.xgboost_model import MODEL_NAME as XGB_NAME, xgb_rmse, xgb_mae, xgb_r2

lr_stats = [lr_rmse, lr_mae, lr_r2]
ridge_stats = [ridge_rmse, ridge_mae, ridge_r2]
rf_stats = [rf_rmse, rf_mae, rf_r2]
xgb_stats = [xgb_rmse, xgb_mae, xgb_r2]
visualize_model_comparison(LR_NAME, lr_stats, RR_NAME, ridge_stats, RF_NAME, rf_stats, XGB_NAME, xgb_stats)
