import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 13, 'display.max_colwidth', 11,
              'display.min_rows', 20, 'display.max_rows', 30, 'display.width', 1000)

# Function to calculate he weighted average of a number Series without NaN values
def average_for_number_column(Series):
    return(sum(Series)/len(Series))

# Function to calculate the weighted average of a string Series without NaN values, reindexed with first integers
def weighted_average_for_string_column(Series):
    Series_value_counts = Series.value_counts()
    Series_value_counts = Series_value_counts.reset_index(drop = True)
    sum_prod = 0
    for i in range(len(Series_value_counts)):
        sum_prod += Series_value_counts.iloc[i] * i
    weighted_average = sum_prod / sum(Series_value_counts)
    return weighted_average

# Load the data, and separate the target y
train_path = './input/train.csv'
train_data = pd.read_csv(train_path)
y = train_data.SalePrice

# Create X
features_selected = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
len_all_features = len(features_selected)

# Select columns corresponding to features
X = train_data[features_selected]

# Warning some features columns contain string values. We have to convert them to numbers.
# Avoid warning when changing values of dataframe:
pd.options.mode.chained_assignment = None

### Replace NAN by average in columns who contain number values
# Select number columns
number_features_selected = [column_name for column_name in features_selected if (X.loc[:, column_name].dtype == 'int64' or X.loc[:, column_name].dtype == 'float64')]
print(number_features_selected)
# Fill columns
# Function to fill 'NaN' entries by average of the other values in numbers columns
def fill_NaN_by_average_in_all_numbers_columns(X, number_features_selected):
    for column_name in number_features_selected:
        # Copy the column without NaN entries before changes to calculate the weighted average later
        column_without_NaN_entries = X.loc[:, column_name].dropna()
        X.loc[:, column_name] = X.loc[:, column_name].fillna(average_for_number_column(column_without_NaN_entries) * 1)
fill_NaN_by_average_in_all_numbers_columns(X, number_features_selected)
### Replace string by numbers in columns who contain object values
# Select object columns
string_features_selected = [column_name for column_name in features_selected if X.loc[:, column_name].dtype == object]
print(string_features_selected)
# Function to convert columns in a DataFrame who contain string values using directories and save these
# For each use, this one save dictionary used into the list below in order to reuse further
dicts = {}
def convert_to_numbers_and_save_dict(column_name, DataFrame, dicts):
    # Fill "NaN" entries with "Unknown" and calculate the weighted average
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].fillna("Unknown")
    column = DataFrame.loc[:, column_name]
    column_value_counts = column.value_counts()
    weighted_average_for_str_column = weighted_average_for_string_column(column) * 1
    # Warning. We have to convert column_value_counts_index into list to find the place of each string value in its index:
    column_value_counts_index = column_value_counts.index.tolist()
    column_dict = {string_value: column_value_counts_index.index(string_value) for string_value in column_value_counts_index}
    # We assure the key "Unknown" is in dictionary to avoid empty entries problems with test_data during submission
    # We give to the "Unknown" key the value of weighted average of values in the new dictionary
    print(column_dict)
    # Associate "Unknown" key with the weighted average value in column_dict
    column_dict["Unknown"] = weighted_average_for_str_column
    print(column_dict)
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].map(lambda p: column_dict[p])
    dicts[column_name] = column_dict

# Convert columns
for column_name in string_features_selected:
    convert_to_numbers_and_save_dict(column_name, X, dicts)
print(dicts)

# Split X and y into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define and fit a random forest model, make validation predictions and calculate mean absolute error
model_default = RandomForestRegressor(random_state=1)
model_default.fit(train_X, train_y)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("\nValidation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(default_val_mae))

### Recalculate MAE for a new list of dicts used
# Functions used
# Function to convert a string columns with a new 'dicts'
def convert_string_column_into_numbers_using_dict_in_dicts(column_name, DataFrame, new_dict):
    # Fill "NaN" entries with "Unknown"
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].fillna("Unknown")
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].map(lambda p: new_dict[p])
# Function to convert all string columns with a new 'list_of_dict_used'
def convert_string_columns_into_numbers_using_dicts(DataFrame, new_dicts, new_string_features_selected):
    i = 0
    for column_name in new_string_features_selected:
        convert_string_column_into_numbers_using_dict_in_dicts(column_name, DataFrame, new_dicts[column_name])
        i += 1
# Function to recalculate MAE with a new dicts and new features_selected
def calculate_MAE(new_features_selected, new_dicts, new_string_features_selected, new_number_features_selected):
    # Restart at the beginning for X
    X = train_data[new_features_selected]
    # Fill columns
    fill_NaN_by_average_in_all_numbers_columns(X, new_number_features_selected)
    # Convert columns with the new dict 'list_of_dict_used'
    convert_string_columns_into_numbers_using_dicts(X, new_dicts, new_string_features_selected)
    # Split X and y into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    # Define and fit a random forest model, make validation predictions and calculate mean absolute error
    model_default = RandomForestRegressor(random_state=1)
    model_default.fit(train_X, train_y)
    model_default_val_preds = model_default.predict(val_X)
    val_mae = mean_absolute_error(model_default_val_preds, val_y)
    #print("\nNew validation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(val_mae))
    return(val_mae)

# Pull out features in features selected to obtain a better MAE
old_val_mae = default_val_mae
features_selected_temp = features_selected
dicts_temp = dicts
string_features_selected_temp = string_features_selected
number_features_selected_temp = number_features_selected
all_features_to_test = features_selected
for feature in all_features_to_test:
    print(feature, 'tested')
    features_selected_temp = features_selected.copy()
    features_selected_temp.remove(feature)
    if feature in string_features_selected:
        dicts_temp = dicts.copy()
        dicts_temp.pop(feature)
        string_features_selected_temp = string_features_selected.copy()
        string_features_selected_temp.remove(feature)
    else:
        number_features_selected_temp = number_features_selected.copy()
        number_features_selected_temp.remove(feature)
    print(string_features_selected_temp)
    new_val_mae = calculate_MAE(features_selected_temp, dicts_temp, string_features_selected_temp, number_features_selected_temp)
    if new_val_mae < old_val_mae:
        features_selected.remove(feature)
        if feature in string_features_selected:
            dicts.pop(feature)
            string_features_selected.remove(feature)
        else:
            number_features_selected.remove(feature)
        old_val_mae = new_val_mae
    else:
        dicts_temp = dicts
        string_features_selected_temp = string_features_selected
        number_features_selected_temp = number_features_selected
    print("New validation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(new_val_mae))

final_val_mae = calculate_MAE(features_selected, dicts, string_features_selected, number_features_selected)
print('features_selected', len(features_selected), '/', len_all_features)
print("final validation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(final_val_mae))

# Fit Model Using All Data
X = train_data[features_selected]
# Fill and convert columns selected as before
fill_NaN_by_average_in_all_numbers_columns(X, number_features_selected)
convert_string_columns_into_numbers_using_dicts(X, dicts, string_features_selected)
print(X.head())
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)

# Get "test" data to make final prediction
test_path = './input/test.csv'
test_data = pd.read_csv(test_path)
test_X = test_data[features_selected]
# Fill and convert columns selected as before
fill_NaN_by_average_in_all_numbers_columns(test_X, number_features_selected)
convert_string_columns_into_numbers_using_dicts(test_X, dicts, string_features_selected)
print(test_X.head())
# make predictions which we will submit.
test_preds = final_model.predict(test_X)

# Generate a submission
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission9.7.csv', index=False)