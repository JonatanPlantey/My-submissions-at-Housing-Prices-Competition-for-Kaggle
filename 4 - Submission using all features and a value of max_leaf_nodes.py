import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 13, 'display.max_colwidth', 11,
              'display.min_rows', 20, 'display.max_rows', 30, 'display.width', 1000)

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
# Select columns corresponding to features
X = train_data[features_selected]

# Warning some features columns contain string values. We have to convert them to numbers.
# Avoid warning when changing values of dataframe:
pd.options.mode.chained_assignment = None

### Replace NAN by 0 in columns who contain number values
# Select number columns
number_features_selected = [column_name for column_name in features_selected if (X.loc[:, column_name].dtype == 'int64' or X.loc[:, column_name].dtype == 'float64')]
print(number_features_selected)
# Fill columns
for column_name in number_features_selected:
    X.loc[:, column_name] = X.loc[:, column_name].fillna(0)

### Replace string by numbers in columns who contain object values
# Select object columns
string_features_selected = [column_name for column_name in features_selected if X.loc[:, column_name].dtype == object]
print(string_features_selected)
# Function to convert columns in a DataFrame who contain string values
def convert_to_numbers(column_name, DataFrame):
    # Fill "nan" entries with "Unknown"
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].fillna("Unknown")
    column_value_counts = DataFrame.loc[:, column_name].value_counts()
    #print(column_value_counts)
    # Warning. We have to convert column_value_counts_index into list to find de place of each string value in its index:
    column_value_counts_index = column_value_counts.index.tolist()
    column_dict = {string_value: column_value_counts_index.index(string_value) for string_value in column_value_counts_index}
    print(column_dict)
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].map(lambda p: column_dict[p])
    return
# Convert columns
for column_name in string_features_selected:
    convert_to_numbers(column_name, X)

# Split X and y into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define and fit a random forest model, make validation predictions and calculate mean absolute error
model_default = RandomForestRegressor(random_state=1)
model_default.fit(train_X, train_y)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("Validation MAE for Model with no value of max_leaf_nodes: {:,.0f}".format(default_val_mae))

# Find and use a good value of max_leaf_nodes
# Function to return MAE for a selected value of max_leaf_nodes:
def get_mae(max_lf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_lf_nodes, random_state=1)
    model.fit(train_X, train_y)
    val_preds = model.predict(val_X)
    val_mae = mean_absolute_error(val_y, val_preds)
    return(val_mae)

# Write loop to find the ideal tree size from candidates_max_leaf_nodes
candidates_max_leaf_nodes = range(300,350)
# This methode use dict comprehension (dictionnary comprehension):
scores = {tree_size: get_mae(tree_size, train_X, val_X, train_y, val_y) for tree_size in candidates_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print("A best value of max_leaf_nodes:", best_tree_size)

# Recalculate mean absolute error with best max_leaf_nodes value
model_max_leaf_nodes = RandomForestRegressor(max_leaf_nodes=best_tree_size,random_state=1)
model_max_leaf_nodes.fit(train_X, train_y)
model_max_leaf_nodes_val_preds = model_max_leaf_nodes.predict(val_X)
max_leaf_nodes_val_mae = mean_absolute_error(model_max_leaf_nodes_val_preds, val_y)
print("Validation MAE for a best value of max_leaf_nodes: {:,.0f}".format(max_leaf_nodes_val_mae))

# Fit Model Using All Data with
final_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)

# Get "test" data to make final prediction
test_path = './input/test.csv'
test_data = pd.read_csv(test_path)
test_X = test_data[features_selected]
# Fill and convert columns selected as before
for column_name in number_features_selected:
    test_X.loc[:, column_name] = test_X.loc[:, column_name].fillna(0)
for column_name in string_features_selected:
    convert_to_numbers(column_name, test_X)
# make predictions which we will submit.
test_preds = final_model.predict(test_X)

# Generate a submission
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission4.csv', index=False)