import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the data, and separate the target
train_path = './input/train.csv'
train_data = pd.read_csv(train_path)
# Create y
y = train_data.SalePrice
# Create X
features_selected = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features_selected]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
model_default = RandomForestRegressor(random_state=1)
# Fit Model
model_default.fit(train_X, train_y)
# Make validation predictions and calculate mean absolute error
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("Validation MAE for Model with no value of max_leaf_nodes: {:,.0f}".format(default_val_mae))

# Find and use a best value of max_leaf_nodes
# Function to return MAE for a selected value of max_leaf_nodes:
def get_mae(max_lf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_lf_nodes, random_state=1)
    model.fit(train_X, train_y)
    val_preds = model.predict(val_X)
    val_mae = mean_absolute_error(val_y, val_preds)
    return(val_mae)

### Step 1: Compare Different Tree Sizes (find the best tree size of model to minimize MAE)
candidate_max_leaf_nodes = [230, 231, 232, 233, 234, 235, 236]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
# This methode use dict comprehension (dictionnary comprehension)
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print("A best value of max_leaf_nodes:", best_tree_size)

## Make new validation predictions and calculate new mean absolute error with a better max_leaf_nodes
# Specify new Model
model_max_leaf_nodes = RandomForestRegressor(max_leaf_nodes=best_tree_size,random_state=1)
# Fit new Model
model_max_leaf_nodes.fit(train_X, train_y)
# Make new validation predictions and calculate new mean absolute error (MAE)
model_max_leaf_nodes_val_preds = model_max_leaf_nodes.predict(val_X)
val_mae = mean_absolute_error(model_max_leaf_nodes_val_preds, val_y)
print("Validation MAE for a best value of max_leaf_nodes: {:,.0f}".format(val_mae))

### Step 2: Fit Model Using All Data
final_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=1)
# fit model on all data from the training data
final_model.fit(X, y)

# Now, read the file of "test" data, and apply your model to make predictions.
test_path = './input/test.csv'
# read test data file using pandas
test_data = pd.read_csv(test_path)
# create test_X which comes from test_data but includes only the columns you used for prediction.
test_X = test_data[features_selected]
# make predictions which we will submit.
test_preds = final_model.predict(test_X)

# Generate a submission
# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission2.csv', index=False)