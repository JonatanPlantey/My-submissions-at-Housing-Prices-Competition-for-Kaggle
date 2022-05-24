import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 13, 'display.max_colwidth', 11,
              'display.min_rows', 20, 'display.max_rows', 30, 'display.width', 1000)

# Avoid warning when changing values of dataframe:
pd.options.mode.chained_assignment = None

# Load the data, and separate the target y
train_path = './input/train.csv'
train_data = pd.read_csv(train_path)
y = train_data.SalePrice
# Create X
features_selected = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
 'MasVnrType','MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

# Select columns corresponding to features
X = train_data[features_selected]

# Select number columns
number_features_selected = [feature for feature in features_selected if (X[feature].dtype == 'int64' or X[feature].dtype == 'float64')]
# Select object columns
string_features_selected = [feature for feature in features_selected if X[feature].dtype == object]
# Select columns to convert
features_to_convert = string_features_selected
features_to_convert.extend(['MSSubClass'])
# Select other column
other_features_selected = [feature for feature in features_selected if feature not in features_to_convert]

# We will need the prices average to fill NaN entries:
average_of_prices = sum(y) / len(y)

# Functions used to calculate average
def average_for_value_in_prices(df, value):
    df_copy = df.copy()
    df_copy.drop( df_copy[ df_copy.iloc[:, 0] != value ].index, inplace=True)
    sum_ = sum(df_copy.iloc[:, 1])
    length = len(df_copy.iloc[:, 1])
    if length > 0:
        return(sum_ / length)
    else:
        return(0)

def average_for_NaN_in_prices(df):
    df_copy = df.copy()
    df_copy = df_copy.fillna("Unknown")
    df_copy.drop( df_copy[ df_copy.iloc[:, 0] != "Unknown" ].index, inplace=True)
    sum_ = sum(df_copy.iloc[:, 1])
    length = len(df_copy.iloc[:, 1])
    if length > 0:
        return(sum_ / length)
    else:
        return(0)

def average_for_NaN_in_feature(df):
    df_copy = df.copy()
    df_copy = df_copy.dropna()
    sum_ = sum(df_copy)
    length = len(df_copy)
    if length > 0:
        return(sum_ / length)
    else:
        return(0)

# Create dictionnaries
dicts = {}
for feature in features_to_convert:
    feature_value_counts = X[feature].value_counts()
    number_of_values = len(feature_value_counts)
    feature_and_prices = pd.concat([X[feature], y], axis=1)
    dict_feature = {feature_value_counts.index[i] : average_for_value_in_prices(feature_and_prices, feature_value_counts.index[i])
                    for i in range(len(feature_value_counts))}
    dict_feature["Unknown"] = average_for_NaN_in_prices(feature_and_prices)
    dicts[feature] = dict_feature

print(dicts)

# Convert values into average of prices in X
def convert_using_saved_dicts(feature, DataFrame, dicts):
    # Fill "NaN" entries with "Unknown"
    DataFrame[feature] = DataFrame[feature].fillna("Unknown")
    DataFrame[feature] = DataFrame[feature].map(lambda p: dicts[feature][p])

for feature in features_to_convert:
    convert_using_saved_dicts(feature, X, dicts)

# Fill NaN entries with average in other features
for feature in other_features_selected:
    X[feature] = X[feature].fillna(average_for_NaN_in_feature(X[feature]))

print(X)

# Split X and y into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define and fit a random forest model, make validation predictions and calculate mean absolute error
model_default = RandomForestRegressor(random_state=1)
model_default.fit(train_X, train_y)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("\nValidation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(default_val_mae))


# Functions used
# Function to search and return the value of nearest key in dict to a value p
def return_value_of_nearest_key_of_p(dict, p):
    if type(p) == str:
        return p
    else:
        list = [key for key in dict]
        if "Unknown" in list:
            list.remove("Unknown")
        list_temp = [np.abs(key - p) for key in list]
        idx = np.argmin(list_temp)
        return list[idx]

# Fit Model Using All Data
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)
final_model_val_preds = final_model.predict(X)
final_model_val_mae = mean_absolute_error(final_model_val_preds, y)
print("\nValidation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(final_model_val_mae))

# Get "test" data to make final prediction
test_path = './input/test.csv'
test_data = pd.read_csv(test_path)
test_X = test_data[features_selected]

# Fill and convert columns selected as before
# Convert values into average of prices in X
def convert_using_nearest_key_in_saved_dicts(feature, DataFrame, dicts):
    # Fill "NaN" entries with "Unknown"
    DataFrame[feature] = DataFrame[feature].fillna("Unknown")
    DataFrame[feature] = DataFrame[feature].map(lambda p: dicts[feature][return_value_of_nearest_key_of_p(dicts[feature], p)])

for feature in features_to_convert:
    convert_using_nearest_key_in_saved_dicts(feature, test_X, dicts)

# Fill NaN entries with average in other features
for feature in other_features_selected:
    test_X[feature] = test_X[feature].fillna(average_for_NaN_in_feature(X[feature]))

print(test_X.head())
# make predictions which we will submit.
test_preds = final_model.predict(test_X)

# Generate a submission
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission14.csv', index=False)