import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
train_path = './input/train.csv'
train_data = pd.read_csv(train_path)
y = train_data.SalePrice

# Create X
features_selected = ['MSSubClass', 'MSZoning', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features_selected]
X.head()

# Use Map function to replace "str" value by a number image noted in a dictionnary
# Avoid warning when changing values of dataframe
pd.options.mode.chained_assignment = None
# Fill "nan" entries with "Unknown"
X.MSZoning = X.MSZoning.fillna("Unknown")
MSZoning_dict = {"Unknown":0, "A":1, "C":2, "C (all)":2, "FV":3, "I":4, "RH":9, "RL":6, "RP":7, "RM":8}
X.MSZoning = X.MSZoning.map(lambda p: MSZoning_dict[p])
# ( or alternately : X.MSZoning = X.loc[:, 'MSZoning'].map(lambda p: MSZoning_dict[p]) )

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model and give prediction and MAE of it
model_default = RandomForestRegressor(random_state=1)
model_default.fit(train_X, train_y)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(default_val_mae))

# Train model on all data for the competition
final_model = RandomForestRegressor(random_state=1)
# fit model on all data from the training data
final_model.fit(X, y)

# Now, read the file of "test" data, and apply your model to make predictions.
train_path = './input/test.csv'
# read test data file using pandas
test_data = pd.read_csv(train_path)
# create test_X which comes from test_data but includes only the columns you used for prediction.
test_X = test_data[features_selected]
test_X.MSZoning = test_X.MSZoning.fillna("Unknown")
test_X.MSZoning = test_X.MSZoning.map(lambda p: MSZoning_dict[p])
# make predictions which we will submit.
test_preds = final_model.predict(test_X)

# Generate a submission
# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission3.csv', index=False)