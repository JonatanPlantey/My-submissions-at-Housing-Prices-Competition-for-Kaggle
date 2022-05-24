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
#features_selected = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
# 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
# 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
# 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
# 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
# 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
# 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
# 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
# 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
# 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

features_selected = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle',
 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold']

# Select columns corresponding to features
X = train_data[features_selected]

# Warning some features columns contain string values. We have to convert them to numbers.
# Avoid warning when changing values of dataframe:
pd.options.mode.chained_assignment = None

### Replace NAN by average in columns who contain number values
# Select number columns
number_features_selected = [column_name for column_name in features_selected if (X.loc[:, column_name].dtype == 'int64' or X.loc[:, column_name].dtype == 'float64')]
#print(number_features_selected)
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
#print(string_features_selected)
# Function to convert columns in a DataFrame who contain string values using directories and save these
# For each use, this one save dictionary used into the list below in order to reuse further
dicts = {}
def convert_to_numbers_and_save_dict(column_name, DataFrame, dicts):
    # Copy the column without NaN entries and calculate the weighted average
    column_without_NaN_entries = DataFrame.loc[:, column_name].dropna(axis = 0)
    column_value_counts = column_without_NaN_entries.value_counts()
    weighted_average_for_str_column = weighted_average_for_string_column(column_without_NaN_entries) * 1
    # Warning. We have to convert column_value_counts_index into list to find the place of each string value in its index:
    column_value_counts_index = column_value_counts.index.tolist()
    column_dict = {string_value: column_value_counts_index.index(string_value) for string_value in column_value_counts_index}
    # We assure the key "Unknown" is in dictionary to avoid empty entries problems with test_data during submission
    # We give to the "Unknown" key the value of weighted average of values in the new dictionary
    #print(column_dict)
    # Fill "NaN" entries with "Unknown"
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].fillna("Unknown")
    # Associate "Unknown" key with the weighted average value in column_dict
    column_dict["Unknown"] = weighted_average_for_str_column
    #print(column_dict)
    DataFrame.loc[:, column_name] = DataFrame.loc[:, column_name].map(lambda p: column_dict[p])
    dicts[column_name] = column_dict

# Convert columns
for column_name in string_features_selected:
    convert_to_numbers_and_save_dict(column_name, X, dicts)
#print(dicts)

# Split X and y into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define and fit a random forest model, make validation predictions and calculate mean absolute error
model_default = RandomForestRegressor(random_state=1)
model_default.fit(train_X, train_y)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("\nValidation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(default_val_mae))

# Adjust dictionnaries of string columns
#dicts = {'MSZoning': {'RL': 0, 'RM': 1, 'FV': 2, 'RH': 3, 'C (all)': 4, 'Unknown': 0.29863013698630136}, 'Street': {'Pave': 0, 'Grvl': 1, 'Unknown': 0.00410958904109589}, 'Alley': {'Grvl': 0, 'Pave': 1, 'Unknown': 0.45054945054945056}, 'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3, 'Unknown': 0.40821917808219177}, 'LandContour': {'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3, 'Unknown': 0.18561643835616437}, 'Utilities': {'AllPub': 0, 'NoSeWa': 1, 'Unknown': 0.0006849315068493151}, 'LotConfig': {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4, 'Unknown': 0.41643835616438357}, 'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2, 'Unknown': 0.06232876712328767}, 'Neighborhood': {'NAmes': 0, 'CollgCr': 1, 'OldTown': 2, 'Edwards': 3, 'Somerst': 4, 'Gilbert': 5, 'NridgHt': 6, 'Sawyer': 7, 'NWAmes': 8, 'SawyerW': 9, 'BrkSide': 10, 'Crawfor': 11, 'Mitchel': 12, 'NoRidge': 13, 'Timber': 14, 'IDOTRR': 15, 'ClearCr': 16, 'StoneBr': 17, 'SWISU': 18, 'MeadowV': 19, 'Blmngtn': 20, 'BrDale': 21, 'Veenker': 22, 'NPkVill': 23, 'Blueste': 24, 'Unknown': 6.6287671232876715}, 'Condition1': {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRAn': 3, 'PosN': 4, 'RRAe': 5, 'PosA': 6, 'RRNn': 7, 'RRNe': 8, 'Unknown': 0.3321917808219178}, 'Condition2': {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRNn': 3, 'PosN': 4, 'PosA': 5, 'RRAn': 6, 'RRAe': 7, 'Unknown': 0.028767123287671233}, 'BldgType': {'1Fam': 0, 'TwnhsE': 1, 'Duplex': 2, 'Twnhs': 3, '2fmCon': 4, 'Unknown': 0.3226027397260274}, 'HouseStyle': {'1Story': 0, '2Story': 1, '1.5Fin': 2, 'SLvl': 3, 'SFoyer': 4, '1.5Unf': 5, '2.5Unf': 6, '2.5Fin': 7, 'Unknown': 0.8821917808219178}, 'RoofStyle': {'Gable': 0, 'Hip': 1, 'Flat': 2, 'Gambrel': 3, 'Mansard': 4, 'Shed': 5, 'Unknown': 0.2623287671232877}, 'RoofMatl': {'CompShg': 0, 'Tar&Grv': 1, 'WdShngl': 2, 'WdShake': 3, 'Metal': 4, 'Membran': 5, 'Roll': 6, 'ClyTile': 7, 'Unknown': 0.0410958904109589}, 'Exterior1st': {'VinylSd': 0, 'HdBoard': 1, 'MetalSd': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CemntBd': 5, 'BrkFace': 6, 'WdShing': 7, 'Stucco': 8, 'AsbShng': 9, 'BrkComm': 10, 'Stone': 11, 'AsphShn': 12, 'ImStucc': 13, 'CBlock': 14, 'Unknown': 2.0273972602739727}, 'Exterior2nd': {'VinylSd': 0, 'MetalSd': 1, 'HdBoard': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CmentBd': 5, 'Wd Shng': 6, 'Stucco': 7, 'BrkFace': 8, 'AsbShng': 9, 'ImStucc': 10, 'Brk Cmn': 11, 'Stone': 12, 'AsphShn': 13, 'Other': 14, 'CBlock': 15, 'Unknown': 2.1794520547945204}, 'MasVnrType': {'None': 0, 'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3, 'Unknown': 0.5137741046831956}, 'ExterQual': {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 0.43424657534246575}, 'ExterCond': {'TA': 0, 'Gd': 1, 'Fa': 2, 'Ex': 3, 'Po': 4, 'Unknown': 0.14726027397260275}, 'Foundation': {'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5, 'Unknown': 0.7102739726027397}, 'BsmtQual': {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 0.6781447645818693}, 'BsmtCond': {'TA': 0, 'Gd': 1, 'Fa': 2, 'Po': 3, 'Unknown': 0.11314125087842586}, 'BsmtExposure': {'No': 0, 'Av': 1, 'Gd': 2, 'Mn': 3, 'Unknown': 0.5843881856540084}, 'BsmtFinType1': {'Unf': 0, 'GLQ': 1, 'ALQ': 2, 'BLQ': 3, 'Rec': 4, 'LwQ': 5, 'Unknown': 1.5488404778636684}, 'BsmtFinType2': {'Unf': 0, 'Rec': 1, 'LwQ': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5, 'Unknown': 0.2749648382559775}, 'Heating': {'GasA': 0, 'GasW': 1, 'Grav': 2, 'Wall': 3, 'OthW': 4, 'Floor': 5, 'Unknown': 0.03904109589041096}, 'HeatingQC': {'Ex': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4, 'Unknown': 0.7267123287671233}, 'CentralAir': {'Y': 0, 'N': 1, 'Unknown': 0.06506849315068493}, 'Electrical': {'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4, 'Unknown': 0.11034955448937629}, 'KitchenQual': {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 0.6184931506849315}, 'Functional': {'Typ': 0, 'Min2': 1, 'Min1': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Unknown': 0.15616438356164383}, 'FireplaceQu': {'Gd': 0, 'TA': 1, 'Fa': 2, 'Ex': 3, 'Po': 4, 'Unknown': 0.6896103896103896}, 'GarageType': {'Attchd': 0, 'Detchd': 1, 'BuiltIn': 2, 'Basment': 3, 'CarPort': 4, '2Types': 5, 'Unknown': 0.49746192893401014}, 'GarageFinish': {'Unf': 0, 'RFn': 1, 'Fin': 2, 'Unknown': 0.8165337200870196}, 'GarageQual': {'TA': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3, 'Po': 4, 'Unknown': 0.07034082668600435}, 'GarageCond': {'TA': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'Ex': 4, 'Unknown': 0.05946337926033358}, 'PavedDrive': {'Y': 0, 'N': 1, 'P': 2, 'Unknown': 0.10273972602739725}, 'PoolQC': {'Gd': 0, 'Ex': 1, 'Fa': 2, 'Unknown': 0.8571428571428571}, 'Fence': {'MnPrv': 0, 'GdPrv': 1, 'GdWo': 2, 'MnWw': 3, 'Unknown': 0.7117437722419929}, 'MiscFeature': {'Shed': 0, 'Gar2': 1, 'Othr': 2, 'TenC': 3, 'Unknown': 0.16666666666666666}, 'SaleType': {'WD': 0, 'New': 1, 'COD': 2, 'ConLD': 3, 'ConLI': 4, 'ConLw': 5, 'CWD': 6, 'Oth': 7, 'Con': 8, 'Unknown': 0.23356164383561645}, 'SaleCondition': {'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5, 'Unknown': 0.3116438356164384}}

# Old dicts
#dicts = {'MSZoning': {'RL': 0, 'RM': 1, 'FV': 2, 'RH': 3, 'C (all)': 4, 'Unknown': 0.29863013698630136},
# 'Street': {'Pave': 0, 'Grvl': 1, 'Unknown': 0.00410958904109589},
# 'Alley': {'Grvl': 0, 'Pave': 1, 'Unknown': 0.45054945054945056},
# 'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3, 'Unknown': 0.40821917808219177},
# 'LandContour': {'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3, 'Unknown': 0.18561643835616437},
# 'Utilities': {'AllPub': 0, 'NoSeWa': 1, 'Unknown': 0.0006849315068493151},
# 'LotConfig': {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4, 'Unknown': 0.41643835616438357},
# 'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2, 'Unknown': 0.06232876712328767},
# 'Neighborhood': {'NAmes': 0, 'CollgCr': 1, 'OldTown': 2, 'Edwards': 3, 'Somerst': 4, 'Gilbert': 5, 'NridgHt': 6, 'Sawyer': 7, 'NWAmes': 8, 'SawyerW': 9, 'BrkSide': 10, 'Crawfor': 11, 'Mitchel': 12, 'NoRidge': 13, 'Timber': 14, 'IDOTRR': 15, 'ClearCr': 16, 'StoneBr': 17, 'SWISU': 18, 'MeadowV': 19, 'Blmngtn': 20, 'BrDale': 21, 'Veenker': 22, 'NPkVill': 23, 'Blueste': 24, 'Unknown': 6.6287671232876715},
# 'Condition1': {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRAn': 3, 'PosN': 4, 'RRAe': 5, 'PosA': 6, 'RRNn': 7, 'RRNe': 8, 'Unknown': 0.3321917808219178},
# 'Condition2': {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRNn': 3, 'PosN': 4, 'PosA': 5, 'RRAn': 6, 'RRAe': 7, 'Unknown': 0.028767123287671233},
# 'BldgType': {'1Fam': 0, 'TwnhsE': 1, 'Duplex': 2, 'Twnhs': 3, '2fmCon': 4, 'Unknown': 0.3226027397260274},
# 'HouseStyle': {'1Story': 0, '2Story': 1, '1.5Fin': 2, 'SLvl': 3, 'SFoyer': 4, '1.5Unf': 5, '2.5Unf': 6, '2.5Fin': 7, 'Unknown': 0.8821917808219178},
# 'RoofStyle': {'Gable': 0, 'Hip': 1, 'Flat': 2, 'Gambrel': 3, 'Mansard': 4, 'Shed': 5, 'Unknown': 0.2623287671232877},
# 'RoofMatl': {'CompShg': 0, 'Tar&Grv': 1, 'WdShngl': 2, 'WdShake': 3, 'Metal': 4, 'Membran': 5, 'Roll': 6, 'ClyTile': 7, 'Unknown': 0.0410958904109589},
# 'Exterior1st': {'VinylSd': 0, 'HdBoard': 1, 'MetalSd': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CemntBd': 5, 'BrkFace': 6, 'WdShing': 7, 'Stucco': 8, 'AsbShng': 9, 'BrkComm': 10, 'Stone': 11, 'AsphShn': 12, 'ImStucc': 13, 'CBlock': 14, 'Unknown': 2.0273972602739727},
# 'Exterior2nd': {'VinylSd': 0, 'MetalSd': 1, 'HdBoard': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CmentBd': 5, 'Wd Shng': 6, 'Stucco': 7, 'BrkFace': 8, 'AsbShng': 9, 'ImStucc': 10, 'Brk Cmn': 11, 'Stone': 12, 'AsphShn': 13, 'Other': 14, 'CBlock': 15, 'Unknown': 2.1794520547945204},
# 'MasVnrType': {'None': 0, 'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3, 'Unknown': 0.5137741046831956},
# 'ExterQual': {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 0.43424657534246575},
# 'ExterCond': {'TA': 0, 'Gd': 1, 'Fa': 2, 'Ex': 3, 'Po': 4, 'Unknown': 0.14726027397260275},
# 'Foundation': {'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5, 'Unknown': 0.7102739726027397},
# 'BsmtQual': {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 0.6781447645818693},
# 'BsmtCond': {'TA': 0, 'Gd': 1, 'Fa': 2, 'Po': 3, 'Unknown': 0.11314125087842586},
# 'BsmtExposure': {'No': 0, 'Av': 1, 'Gd': 2, 'Mn': 3, 'Unknown': 0.5843881856540084},
# 'BsmtFinType1': {'Unf': 0, 'GLQ': 1, 'ALQ': 2, 'BLQ': 3, 'Rec': 4, 'LwQ': 5, 'Unknown': 1.5488404778636684},
# 'BsmtFinType2': {'Unf': 0, 'Rec': 1, 'LwQ': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5, 'Unknown': 0.2749648382559775},
# 'Heating': {'GasA': 0, 'GasW': 1, 'Grav': 2, 'Wall': 3, 'OthW': 4, 'Floor': 5, 'Unknown': 0.03904109589041096},
# 'HeatingQC': {'Ex': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4, 'Unknown': 0.7267123287671233},
# 'CentralAir': {'Y': 0, 'N': 1, 'Unknown': 0.06506849315068493},
# 'Electrical': {'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4, 'Unknown': 0.11034955448937629},
# 'KitchenQual': {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3, 'Unknown': 0.6184931506849315},
# 'Functional': {'Typ': 0, 'Min2': 1, 'Min1': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Unknown': 0.15616438356164383},
# 'FireplaceQu': {'Gd': 0, 'TA': 1, 'Fa': 2, 'Ex': 3, 'Po': 4, 'Unknown': 0.6896103896103896},
# 'GarageType': {'Attchd': 0, 'Detchd': 1, 'BuiltIn': 2, 'Basment': 3, 'CarPort': 4, '2Types': 5, 'Unknown': 0.49746192893401014},
# 'GarageFinish': {'Unf': 0, 'RFn': 1, 'Fin': 2, 'Unknown': 0.8165337200870196},
# 'GarageQual': {'TA': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3, 'Po': 4, 'Unknown': 0.07034082668600435},
# 'GarageCond': {'TA': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'Ex': 4, 'Unknown': 0.05946337926033358},
# 'PavedDrive': {'Y': 0, 'N': 1, 'P': 2, 'Unknown': 0.10273972602739725},
# 'PoolQC': {'Gd': 0, 'Ex': 1, 'Fa': 2, 'Unknown': 0.8571428571428571},
# 'Fence': {'MnPrv': 0, 'GdPrv': 1, 'GdWo': 2, 'MnWw': 3, 'Unknown': 0.7117437722419929},
# 'MiscFeature': {'Shed': 0, 'Gar2': 1, 'Othr': 2, 'TenC': 3, 'Unknown': 0.16666666666666666},
# 'SaleType': {'WD': 0, 'New': 1, 'COD': 2, 'ConLD': 3, 'ConLI': 4, 'ConLw': 5, 'CWD': 6, 'Oth': 7, 'Con': 8, 'Unknown': 0.23356164383561645},
# 'SaleCondition': {'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5, 'Unknown': 0.3116438356164384}}

# New dicts
dicts = {'MSZoning': {'RL': 3, 'RM': 1, 'FV': 4, 'RH': 2, 'C (all)': 0, 'Unknown': 0.29863013698630136},
 'Street': {'Pave': 1, 'Grvl': 0, 'Unknown': 0.00410958904109589},
 'Alley': {'Grvl': 0, 'Pave': 2, 'NA':1, 'Unknown': 0.45054945054945056},
 'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 3, 'IR3': 2, 'Unknown': 0.40821917808219177},
 'LandContour': {'Lvl': 1, 'Bnk': 0, 'HLS': 3, 'Low': 2, 'Unknown': 0.18561643835616437},
 'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'Unknown': 0.0006849315068493151},
 'LotConfig': {'Inside': 0, 'Corner': 1, 'CulDSac': 4, 'FR2': 2, 'FR3': 3, 'Unknown': 0.41643835616438357},
 'LandSlope': {'Gtl': 0, 'Mod': 2, 'Sev': 1, 'Unknown': 0.06232876712328767},
 'Neighborhood': {'NAmes': 9, 'CollgCr': 16, 'OldTown': 1, 'Edwards': 8, 'Somerst': 23, 'Gilbert': 13, 'NridgHt': 22, 'Sawyer': 11, 'NWAmes': 12, 'SawyerW': 17, 'BrkSide': 6, 'Crawfor': 18, 'Mitchel': 14, 'NoRidge': 24, 'Timber': 19, 'IDOTRR': 2, 'ClearCr': 20, 'StoneBr': 3, 'SWISU': 10, 'MeadowV': 4, 'Blmngtn': 15, 'BrDale': 0, 'Veenker': 21, 'NPkVill': 7, 'Blueste': 5, 'Unknown': 6.6287671232876715},
 'Condition1': {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRAn': 3, 'PosN': 4, 'RRAe': 5, 'PosA': 6, 'RRNn': 7, 'RRNe': 8, 'Unknown': 0.3321917808219178},
 'Condition2': {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRNn': 3, 'PosN': 4, 'PosA': 5, 'RRAn': 6, 'RRAe': 7, 'Unknown': 0.028767123287671233},
 'BldgType': {'1Fam': 3, 'TwnhsE': 4, 'Duplex': 1, 'Twnhs': 2, '2fmCon': 0, 'Unknown': 0.3226027397260274},
 'HouseStyle': {'1Story': 4, '2Story': 6, '1.5Fin': 1, 'SLvl': 5, 'SFoyer': 3, '1.5Unf': 0, '2.5Unf': 2, '2.5Fin': 7, 'Unknown': 0.8821917808219178},
 'RoofStyle': {'Gable': 2, 'Hip': 1, 'Flat': 4, 'Gambrel': 0, 'Mansard': 3, 'Shed': 5, 'Unknown': 0.2623287671232877},
 'RoofMatl': {'CompShg': 2, 'Tar&Grv': 3, 'WdShngl': 7, 'WdShake': 6, 'Metal': 4, 'Membran': 5, 'Roll': 0, 'ClyTile': 1, 'Unknown': 0.0410958904109589},
 'Exterior1st': {'VinylSd': 0, 'HdBoard': 1, 'MetalSd': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CemntBd': 5, 'BrkFace': 6, 'WdShing': 7, 'Stucco': 8, 'AsbShng': 9, 'BrkComm': 10, 'Stone': 11, 'AsphShn': 12, 'ImStucc': 13, 'CBlock': 14, 'Unknown': 2.0273972602739727},
 'Exterior2nd': {'VinylSd': 0, 'MetalSd': 1, 'HdBoard': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CmentBd': 5, 'Wd Shng': 6, 'Stucco': 7, 'BrkFace': 8, 'AsbShng': 9, 'ImStucc': 10, 'Brk Cmn': 11, 'Stone': 12, 'AsphShn': 13, 'Other': 14, 'CBlock': 15, 'Unknown': 2.1794520547945204},
 'MasVnrType': {'None': 1, 'BrkFace': 2, 'Stone': 3, 'BrkCmn': 0, 'Unknown': 0.5137741046831956},
 'ExterQual': {'TA': 2, 'Gd': 3, 'Ex': 4, 'Fa': 1, 'Po':0, 'Unknown': 0.43424657534246575},
 'ExterCond': {'TA': 2, 'Gd': 3, 'Fa': 1, 'Ex': 4, 'Po': 0, 'Unknown': 0.14726027397260275},
 'Foundation': {'PConc': 5, 'CBlock': 3, 'BrkTil': 1, 'Slab': 0, 'Stone': 2, 'Wood': 4, 'Unknown': 0.7102739726027397},
 'BsmtQual': {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2, 'Po':0, 'Unknown': 0.6781447645818693},
 'BsmtCond': {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2, 'Po':0, 'Unknown': 0.11314125087842586},
 'BsmtExposure': {'No': 1, 'Av': 3, 'Gd': 4, 'Mn': 2, 'Unknown': 0.5843881856540084},
 'BsmtFinType1': {'Unf': 5, 'GLQ': 6, 'ALQ': 4, 'BLQ': 2, 'Rec': 3, 'LwQ': 1, 'Unknown': 1.5488404778636684},
 'BsmtFinType2': {'Unf': 4, 'Rec': 2, 'LwQ': 3, 'BLQ': 1, 'ALQ': 5, 'GLQ': 6, 'Unknown': 0.2749648382559775},
 'Heating': {'GasA': 5, 'GasW': 4, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'Floor': 0, 'Unknown': 0.03904109589041096},
 'HeatingQC': {'Ex': 4, 'TA': 2, 'Gd': 3, 'Fa': 1, 'Po': 0, 'Unknown': 0.7267123287671233},
 'CentralAir': {'Y': 1, 'N': 0, 'Unknown': 0.06506849315068493},
 'Electrical': {'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0, 'Unknown': 0.11034955448937629},
 'KitchenQual': {'TA': 2, 'Gd': 3, 'Ex': 4, 'Fa': 1, 'Po': 0, 'Unknown': 0.6184931506849315},
 'Functional': {'Typ': 6, 'Min2': 4, 'Min1': 3, 'Mod': 2, 'Maj1': 5, 'Maj2': 0, 'Sev': 1, 'Unknown': 0.15616438356164383},
 'FireplaceQu': {'Gd': 4, 'TA': 3, 'Fa': 2, 'Ex': 5, 'Po': 1, 'NA':0, 'Unknown': 0.6896103896103896},
 'GarageType': {'Attchd': 5, 'Detchd': 2, 'BuiltIn': 6, 'Basment': 3, 'CarPort': 1, '2Types': 4, 'Unknown': 0.49746192893401014},
 'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3, 'NA':0, 'Unknown': 0.8165337200870196},
 'GarageQual': {'TA': 3, 'Fa': 2, 'Gd': 4, 'Ex': 5, 'Po': 1, 'NA':0, 'Unknown': 0.07034082668600435},
 'GarageCond': {'TA': 3, 'Fa': 2, 'Gd': 4, 'Po': 1, 'Ex': 5, 'NA':0, 'Unknown': 0.05946337926033358},
 'PavedDrive': {'Y': 2, 'N': 0, 'P': 1, 'Unknown': 0.10273972602739725},
 'PoolQC': {'Gd': 3, 'Ex': 4, 'Fa': 1, 'TA':2, 'NA':0, 'Unknown': 0.8571428571428571},
 'Fence': {'MnPrv': 1, 'GdPrv': 2, 'GdWo': 2, 'MnWw': 1, 'NA':0, 'Unknown': 0.7117437722419929},
 'MiscFeature': {'Shed': 2, 'Gar2': 3, 'Othr': 1, 'TenC': 4, 'Elev':5, 'NA':0, 'Unknown': 0.16666666666666666},
 'SaleType': {'WD': 0, 'New': 1, 'COD': 2, 'ConLD': 3, 'ConLI': 4, 'ConLw': 5, 'CWD': 6, 'Oth': 7, 'Con': 8, 'Unknown': 0.23356164383561645},
 'SaleCondition': {'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5, 'Unknown': 0.3116438356164384}}
# We add the 'MSSubClass' to strings column to change its values
dicts['MSSubClass'] = {20: 10, 30: 1, 40: 7, 45: 2, 50: 4, 60: 14, 70: 9, 75: 11, 80: 12,
                       85: 6, 90: 5, 120: 13, 150: 10, 160: 8, 180: 0, 190: 3}
string_features_selected = ['MSSubClass', *string_features_selected]

# Recalculate average in new dicts for 'Unknown' values
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
def fill_NaN_by_average_in_all_strings_columns_save_it_to_dicts(X, string_features_selected, dicts):
    for column_name in string_features_selected:
        # Copy the column without NaN entries before changes to calculate the weighted average later
        column_without_NaN_entries = X.loc[:, column_name].dropna()
        column_average = average_for_number_column(column_without_NaN_entries) * 1
        X.loc[:, column_name] = X.loc[:, column_name].fillna(column_average)
        dicts[column_name]["Unknown"] = column_average
        print(dicts[column_name])

# Restart with a new copy of X
X = train_data[features_selected]
# Fill and convert columns selected as before
fill_NaN_by_average_in_all_numbers_columns(X, number_features_selected)
convert_string_columns_into_numbers_using_dicts(X, dicts, string_features_selected)
# Each string column is now a number column, so we can replace NaN entries by average calculated in number column
fill_NaN_by_average_in_all_strings_columns_save_it_to_dicts(X, string_features_selected, dicts)


### Recalculate MAE with new dicts
# Restart with a new copy of X
X = train_data[features_selected]
# Fill and convert columns selected as before
fill_NaN_by_average_in_all_numbers_columns(X, number_features_selected)
convert_string_columns_into_numbers_using_dicts(X, dicts, string_features_selected)
# Split X and y into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Define and fit a random forest model, make validation predictions and calculate mean absolute error
model_default = RandomForestRegressor(random_state=1)
model_default.fit(train_X, train_y)
model_default_val_preds = model_default.predict(val_X)
default_val_mae = mean_absolute_error(model_default_val_preds, val_y)
print("\nValidation MAE for Model with no value of max_leaf_node: {:,.0f}\n".format(default_val_mae))


# Fit Model Using All Data
# Restart with a new copy of X
X = train_data[features_selected]
# Fill and convert columns selected as before
fill_NaN_by_average_in_all_numbers_columns(X, number_features_selected)
convert_string_columns_into_numbers_using_dicts(X, dicts, string_features_selected)
#print(X.head())
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)

# Get "test" data to make final prediction
test_path = './input/test.csv'
test_data = pd.read_csv(test_path)
test_X = test_data[features_selected]
# Fill and convert columns selected as before
fill_NaN_by_average_in_all_numbers_columns(test_X, number_features_selected)
convert_string_columns_into_numbers_using_dicts(test_X, dicts, string_features_selected)
#print(test_X.head())
# make predictions which we will submit.
test_preds = final_model.predict(test_X)

# Generate a submission
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission11.csv', index=False)