import yaml
import pandas as pd


with open('data_processor/data_configuration.yaml', 'r') as file:
    data = yaml.safe_load(file)


# To check csv dataset for null values and other information
def print_data_frame_information(data_frame):
    data_frame.info()


store_sales = pd.read_csv(data['FILE_TO_READ'])

store_sales = store_sales.drop(
    data['DROPPED_COLUMNS'], axis=data['AXIS_COLUMN'])
store_sales[data['DATE_COLUMN']] = pd.to_datetime(
    store_sales[data['DATE_COLUMN']])
store_sales[data['DATE_COLUMN']] = store_sales[data['DATE_COLUMN']
                                               ].dt.to_period(data['PERIOD_TYPE'])

monthly_sales = store_sales.groupby(data['DATE_COLUMN']).sum().reset_index()
monthly_sales[data['DATE_COLUMN']
              ] = monthly_sales[data['DATE_COLUMN']].dt.to_timestamp()
monthly_sales[data['SALES_DIFFERENCE_COLUMN']
              ] = monthly_sales[data['SALES_COLUMN']].diff()
monthly_sales = monthly_sales.dropna()

# Supervised data preparation for models
supervised_data = monthly_sales.drop(
    [data['DATE_COLUMN'], data['SALES_COLUMN']], axis=data['AXIS_COLUMN'])

for i in range(data['START_MONTH_FOR_LOOP'], data['END_MONTH_FOR_LOOP']):
    col_name = data['MONTH_COLUMN'] + str(i)
    supervised_data[col_name] = supervised_data[data['SALES_DIFFERENCE_COLUMN']].shift(
        i)

supervised_data = supervised_data.dropna().reset_index(drop=True)

sales_dates = monthly_sales[data['DATE_COLUMN']
                            ][data['PREDICTION_PERIOD']:].reset_index(drop=True)

act_sales = monthly_sales[data['SALES_COLUMN']
                          ][data['SALE_VALUES_PERIOD']:].to_list()
