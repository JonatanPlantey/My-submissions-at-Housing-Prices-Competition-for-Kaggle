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
        #sum_prod += Series_value_counts.iloc[i] * i
        sum_prod += Series_value_counts.iloc[i] * (i + 1)
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
list_of_dict_used = []
def convert_to_numbers_and_save_dict_used(column_name, DataFrame, list_of_dict):
    # Copy the column without NaN entries before changes to calculate the weighted average later
    column_without_NaN_entries = DataFrame.loc[:, column_name].dropna()
    # Fill "NaN" entries with "Unknown"
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].fillna("Unknown")
    column_value_counts = DataFrame.loc[:, column_name].value_counts()
    #print(column_value_counts)
    # Warning. We have to convert column_value_counts_index into list to find the place of each string value in its index:
    column_value_counts_index = column_value_counts.index.tolist()
    column_dict = {string_value: column_value_counts_index.index(string_value) for string_value in column_value_counts_index}
    # We assure the key "Unknown" is in dictionary to avoid empty entries problems with test_data during submission
    # We give to the "Unknown" key the value of weighted average of values in the new dictionary
    print(column_dict)
    # Calculate the weighted average
    column_dict["Unknown"] = weighted_average_for_string_column(column_without_NaN_entries) * 1
    print(column_dict)
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].map(lambda p: column_dict[p])
    list_of_dict = list_of_dict.append(column_dict)

# Convert columns
for column_name in string_features_selected:
    convert_to_numbers_and_save_dict_used(column_name, X, list_of_dict_used)

# Split X and y into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define and fit a random forest model, make validation predictions and calculate mean absolute error
model_default = RandomForestRegressor(random_state=1)
model_default.fit(train_X, train_y)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("\nValidation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(default_val_mae))

# Adjust dictionnaries of string columns
{'RL': 0, 'RM': 1, 'FV': 2, 'RH': 3, 'C (all)': 4}
{'RL': 0, 'RM': 1, 'FV': 2, 'RH': 3, 'C (all)': 4, 'Unknown': 1.2986301369863014}
{'Pave': 0, 'Grvl': 1}
{'Pave': 0, 'Grvl': 1, 'Unknown': 1.0041095890410958}
{'Unknown': 0, 'Grvl': 1, 'Pave': 2}
{'Unknown': 1.4505494505494505, 'Grvl': 1, 'Pave': 2}
{'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
{'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3, 'Unknown': 1.4082191780821918}
{'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3}
{'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3, 'Unknown': 1.1856164383561645}
{'AllPub': 0, 'NoSeWa': 1}
{'AllPub': 0, 'NoSeWa': 1, 'Unknown': 1.0006849315068493}
{'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4}
{'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4, 'Unknown': 1.4164383561643836}
{'Gtl': 0, 'Mod': 1, 'Sev': 2}
{'Gtl': 0, 'Mod': 1, 'Sev': 2, 'Unknown': 1.0623287671232877}
{'NAmes': 0, 'CollgCr': 1, 'OldTown': 2, 'Edwards': 3, 'Somerst': 4, 'Gilbert': 5, 'NridgHt': 6, 'Sawyer': 7, 'NWAmes': 8, 'SawyerW': 9, 'BrkSide': 10, 'Crawfor': 11, 'Mitchel': 12, 'NoRidge': 13, 'Timber': 14, 'IDOTRR': 15, 'ClearCr': 16, 'StoneBr': 17, 'SWISU': 18, 'MeadowV': 19, 'Blmngtn': 20, 'BrDale': 21, 'Veenker': 22, 'NPkVill': 23, 'Blueste': 24}
{'NAmes': 0, 'CollgCr': 1, 'OldTown': 2, 'Edwards': 3, 'Somerst': 4, 'Gilbert': 5, 'NridgHt': 6, 'Sawyer': 7, 'NWAmes': 8, 'SawyerW': 9, 'BrkSide': 10, 'Crawfor': 11, 'Mitchel': 12, 'NoRidge': 13, 'Timber': 14, 'IDOTRR': 15, 'ClearCr': 16, 'StoneBr': 17, 'SWISU': 18, 'MeadowV': 19, 'Blmngtn': 20, 'BrDale': 21, 'Veenker': 22, 'NPkVill': 23, 'Blueste': 24, 'Unknown': 7.6287671232876715}
{'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRAn': 3, 'PosN': 4, 'RRAe': 5, 'PosA': 6, 'RRNn': 7, 'RRNe': 8}
{'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRAn': 3, 'PosN': 4, 'RRAe': 5, 'PosA': 6, 'RRNn': 7, 'RRNe': 8, 'Unknown': 1.332191780821918}
{'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRNn': 3, 'PosN': 4, 'PosA': 5, 'RRAn': 6, 'RRAe': 7}
{'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRNn': 3, 'PosN': 4, 'PosA': 5, 'RRAn': 6, 'RRAe': 7, 'Unknown': 1.0287671232876712}
{'1Fam': 0, 'TwnhsE': 1, 'Duplex': 2, 'Twnhs': 3, '2fmCon': 4}
{'1Fam': 0, 'TwnhsE': 1, 'Duplex': 2, 'Twnhs': 3, '2fmCon': 4, 'Unknown': 1.3226027397260274}
{'1Story': 0, '2Story': 1, '1.5Fin': 2, 'SLvl': 3, 'SFoyer': 4, '1.5Unf': 5, '2.5Unf': 6, '2.5Fin': 7}
{'1Story': 0, '2Story': 1, '1.5Fin': 2, 'SLvl': 3, 'SFoyer': 4, '1.5Unf': 5, '2.5Unf': 6, '2.5Fin': 7, 'Unknown': 1.8821917808219177}
{'Gable': 0, 'Hip': 1, 'Flat': 2, 'Gambrel': 3, 'Mansard': 4, 'Shed': 5}
{'Gable': 0, 'Hip': 1, 'Flat': 2, 'Gambrel': 3, 'Mansard': 4, 'Shed': 5, 'Unknown': 1.2623287671232877}
{'CompShg': 0, 'Tar&Grv': 1, 'WdShngl': 2, 'WdShake': 3, 'Metal': 4, 'Membran': 5, 'Roll': 6, 'ClyTile': 7}
{'CompShg': 0, 'Tar&Grv': 1, 'WdShngl': 2, 'WdShake': 3, 'Metal': 4, 'Membran': 5, 'Roll': 6, 'ClyTile': 7, 'Unknown': 1.0410958904109588}
{'VinylSd': 0, 'HdBoard': 1, 'MetalSd': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CemntBd': 5, 'BrkFace': 6, 'WdShing': 7, 'Stucco': 8, 'AsbShng': 9, 'BrkComm': 10, 'Stone': 11, 'AsphShn': 12, 'ImStucc': 13, 'CBlock': 14}
{'VinylSd': 0, 'HdBoard': 1, 'MetalSd': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CemntBd': 5, 'BrkFace': 6, 'WdShing': 7, 'Stucco': 8, 'AsbShng': 9, 'BrkComm': 10, 'Stone': 11, 'AsphShn': 12, 'ImStucc': 13, 'CBlock': 14, 'Unknown': 3.0273972602739727}
{'VinylSd': 0, 'MetalSd': 1, 'HdBoard': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CmentBd': 5, 'Wd Shng': 6, 'Stucco': 7, 'BrkFace': 8, 'AsbShng': 9, 'ImStucc': 10, 'Brk Cmn': 11, 'Stone': 12, 'AsphShn': 13, 'Other': 14, 'CBlock': 15}
{'VinylSd': 0, 'MetalSd': 1, 'HdBoard': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CmentBd': 5, 'Wd Shng': 6, 'Stucco': 7, 'BrkFace': 8, 'AsbShng': 9, 'ImStucc': 10, 'Brk Cmn': 11, 'Stone': 12, 'AsphShn': 13, 'Other': 14, 'CBlock': 15, 'Unknown': 3.1794520547945204}
{'None': 0, 'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3, 'Unknown': 4}
{'None': 0, 'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3, 'Unknown': 1.5137741046831956}
{'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3}
{'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 1.4342465753424658}
{'TA': 0, 'Gd': 1, 'Fa': 2, 'Ex': 3, 'Po': 4}
{'TA': 0, 'Gd': 1, 'Fa': 2, 'Ex': 3, 'Po': 4, 'Unknown': 1.1472602739726028}
{'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5}
{'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5, 'Unknown': 1.7102739726027398}
{'TA': 0, 'Gd': 1, 'Ex': 2, 'Unknown': 3, 'Fa': 4}
{'TA': 0, 'Gd': 1, 'Ex': 2, 'Unknown': 1.6781447645818692, 'Fa': 4}
{'TA': 0, 'Gd': 1, 'Fa': 2, 'Unknown': 3, 'Po': 4}
{'TA': 0, 'Gd': 1, 'Fa': 2, 'Unknown': 1.113141250878426, 'Po': 4}
{'No': 0, 'Av': 1, 'Gd': 2, 'Mn': 3, 'Unknown': 4}
{'No': 0, 'Av': 1, 'Gd': 2, 'Mn': 3, 'Unknown': 1.5843881856540085}
{'Unf': 0, 'GLQ': 1, 'ALQ': 2, 'BLQ': 3, 'Rec': 4, 'LwQ': 5, 'Unknown': 6}
{'Unf': 0, 'GLQ': 1, 'ALQ': 2, 'BLQ': 3, 'Rec': 4, 'LwQ': 5, 'Unknown': 2.5488404778636684}
{'Unf': 0, 'Rec': 1, 'LwQ': 2, 'Unknown': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
{'Unf': 0, 'Rec': 1, 'LwQ': 2, 'Unknown': 1.2749648382559775, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
{'GasA': 0, 'GasW': 1, 'Grav': 2, 'Wall': 3, 'OthW': 4, 'Floor': 5}
{'GasA': 0, 'GasW': 1, 'Grav': 2, 'Wall': 3, 'OthW': 4, 'Floor': 5, 'Unknown': 1.039041095890411}
{'Ex': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4}
{'Ex': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4, 'Unknown': 1.7267123287671233}
{'Y': 0, 'N': 1}
{'Y': 0, 'N': 1, 'Unknown': 1.0650684931506849}
{'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4, 'Unknown': 5}
{'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4, 'Unknown': 1.1103495544893762}
{'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3}
{'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 1.6184931506849316}
{'Typ': 0, 'Min2': 1, 'Min1': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6}
{'Typ': 0, 'Min2': 1, 'Min1': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Unknown': 1.1561643835616437}
{'Unknown': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Ex': 4, 'Po': 5}
{'Unknown': 1.6896103896103896, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Ex': 4, 'Po': 5}
{'Attchd': 0, 'Detchd': 1, 'BuiltIn': 2, 'Unknown': 3, 'Basment': 4, 'CarPort': 5, '2Types': 6}
{'Attchd': 0, 'Detchd': 1, 'BuiltIn': 2, 'Unknown': 1.49746192893401, 'Basment': 4, 'CarPort': 5, '2Types': 6}
{'Unf': 0, 'RFn': 1, 'Fin': 2, 'Unknown': 3}
{'Unf': 0, 'RFn': 1, 'Fin': 2, 'Unknown': 1.8165337200870195}
{'TA': 0, 'Unknown': 1, 'Fa': 2, 'Gd': 3, 'Ex': 4, 'Po': 5}
{'TA': 0, 'Unknown': 1.0703408266860044, 'Fa': 2, 'Gd': 3, 'Ex': 4, 'Po': 5}
{'TA': 0, 'Unknown': 1, 'Fa': 2, 'Gd': 3, 'Po': 4, 'Ex': 5}
{'TA': 0, 'Unknown': 1.0594633792603336, 'Fa': 2, 'Gd': 3, 'Po': 4, 'Ex': 5}
{'Y': 0, 'N': 1, 'P': 2}
{'Y': 0, 'N': 1, 'P': 2, 'Unknown': 1.1027397260273972}
{'Unknown': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3}
{'Unknown': 1.8571428571428572, 'Gd': 1, 'Ex': 2, 'Fa': 3}
{'Unknown': 0, 'MnPrv': 1, 'GdPrv': 2, 'GdWo': 3, 'MnWw': 4}
{'Unknown': 1.7117437722419928, 'MnPrv': 1, 'GdPrv': 2, 'GdWo': 3, 'MnWw': 4}
{'Unknown': 0, 'Shed': 1, 'Gar2': 2, 'Othr': 3, 'TenC': 4}
{'Unknown': 1.1666666666666667, 'Shed': 1, 'Gar2': 2, 'Othr': 3, 'TenC': 4}
{'WD': 0, 'New': 1, 'COD': 2, 'ConLD': 3, 'ConLI': 4, 'ConLw': 5, 'CWD': 6, 'Oth': 7, 'Con': 8}
{'WD': 0, 'New': 1, 'COD': 2, 'ConLD': 3, 'ConLI': 4, 'ConLw': 5, 'CWD': 6, 'Oth': 7, 'Con': 8, 'Unknown': 1.2335616438356165}
{'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5}
{'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5, 'Unknown': 1.3116438356164384}


# Functions used
# Function to convert a string columns with a new 'list_of_dict_used'
def convert_string_column_into_numbers_using_dict_in_list_of_dict_used(column_name, DataFrame, dict):
    # Fill "NaN" entries with "Unknown"
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].fillna("Unknown")
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].map(lambda p: dict[p])
# Function to convert all string columns with a new 'list_of_dict_used'
def convert_string_columns_into_numbers_using_list_of_dict_used(X, list_of_dict_used):
    i = 0
    for column_name in string_features_selected:
        convert_string_column_into_numbers_using_dict_in_list_of_dict_used(column_name, X, list_of_dict_used[i])
        i += 1

# Fit Model Using All Data
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)

# Get "test" data to make final prediction
test_path = './input/test.csv'
test_data = pd.read_csv(test_path)
test_X = test_data[features_selected]
# Fill and convert columns selected as before
fill_NaN_by_average_in_all_numbers_columns(test_X, number_features_selected)
convert_string_columns_into_numbers_using_list_of_dict_used(test_X, list_of_dict_used)
print(test_X.head())
# make predictions which we will submit.
test_preds = final_model.predict(test_X)

# Generate a submission
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission9.csv', index=False)