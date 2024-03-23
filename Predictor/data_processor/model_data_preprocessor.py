import yaml
from sklearn.preprocessing import MinMaxScaler
from data_processor.data_preparator import supervised_data

with open('data_processor/preprocessor_configuration.yaml', 'r') as file:
    process_config = yaml.safe_load(file)

# Splitting dataset
train_data = supervised_data[:process_config['SPLIT_INDEX']]
test_data = supervised_data[process_config['SPLIT_INDEX']:]

# Center features to manageable range
scaler = MinMaxScaler(feature_range=(
    process_config['MIN_FEATURE_VALUE'], process_config['MAX_FEATURE_VALUE']))
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# First column selected to the output , remaining columns selected as input features
x_train, y_train = (train_data[:, process_config['COLUMN_SLICE_END_INDEX']:],
                    train_data[:, process_config['COLUMN_SLICE_START_INDEX']:process_config['COLUMN_SLICE_END_INDEX']])
x_test, y_test = (test_data[:, process_config['COLUMN_SLICE_END_INDEX']:],
                  test_data[:, process_config['COLUMN_SLICE_START_INDEX']:process_config['COLUMN_SLICE_END_INDEX']])
y_train = y_train.ravel()
y_test = y_test.ravel()
