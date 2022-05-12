# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
train_path = './input/train.csv'
train_data = pd.read_csv(train_path)
# Create target object and call it y
y = train_data.SalePrice
# Create X
features_selected = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select columns corresponding to features, and preview the data
X = train_data[features_selected]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model and give prediction and MAE of it
# Specify Model
model_default = RandomForestRegressor(random_state=1)
# Fit Model
model_default.fit(train_X, train_y)
# Make new validation predictions and calculate new mean absolute error (MAE)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("Validation MAE for RandomForestRegressor Model: {:,.0f}".format(default_val_mae))

# Train a model for the competition

# To improve accuracy, create a new Random Forest model which you will train on all training data
final_model = RandomForestRegressor(random_state=1)
# fit rf_model_on_full_data on all data from the training data
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
output.to_csv('submission1.csv', index=False)