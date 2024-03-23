def print_splitting_result(train, test):
    print('Train Data Shape:', train.shape)
    print('Test Data Shape:', test.shape)


def print_train_and_test_data_shape(x_train, y_train, x_test, y_test):
    print('X_train Shape:', x_train.shape)
    print('y_train Shape:', y_train.shape)
    print('X_test Shape:', x_test.shape)
    print('y_test Shape:', y_test.shape)


def print_metrics_calculation(model_type, rmse, mae, r2):
    print('{} RMSE: {}'.format(model_type, rmse))
    print('{} MAE: {}'.format(model_type, mae))
    print('{} R2 Score: {}'.format(model_type, r2))


def print_best_mae(model):
    print("Best MAE: {:.2f} with {} rounds".format(
        model.best_score, model.best_iteration + 1))
