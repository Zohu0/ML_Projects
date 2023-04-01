import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Loading The Data
housing = pd.read_csv('data.csv')

# Analyzing The Data
# print(housing.head())
# print(housing.info())
# print(housing['CHAS'])
# print(housing['CHAS'].value_counts())
# print(housing['LSTAT'].value_counts())
# print(housing.describe())


# for plotting histogram
# %matplotlib inline
# housing.hist(bins = 50, figsize=(20,15))


#Train-Test-Split

# for Understanding puprose
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
# train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {len(train_set)}\n Rows in test set: {len(test_set)}\n")


#Function Is Available In Sklearn
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(f"Rows in train set: {len(train_set)}\n Rows in test set: {len(test_set)}\n")


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


housing = strat_train_set.copy() #important point



#Looking For CoRelation
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# from pandas.plotting import scatter_matrix
# attributes = ['MEDV', "RM", 'ZN', 'LSTAT']
# scatter_matrix(housing[attributes], figsize = (12, 8))
# housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


#Trying Out Attribute Information
housing['TAXRM']= housing['TAX']/housing['RM']
# print(housing.head())

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)

housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Creating PipeLine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

housing_tr = my_pipeline.fit_transform(housing)
# print(housing_tr.shape)


#Selecting A desire Model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_tr, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data) # to see prediction
list(some_labels) # to compare my prediction


#Evaluating The Model
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

print(rmse)


#Using Better Evaluation Technique - Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_tr, housing_labels, scoring = "neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print(rmse_scores)

def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())


print(print_scores(rmse_scores))


#Saving The Model
from joblib import dump, load
dump(model, "Dragon.joblib")


#Testing the model on test data
x_test = strat_test_set.drop("MEDV", axis=1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_prediction= model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_prediction)
final_rmse = np.sqrt(final_mse)

print(final_rmse)

#Using The Model
model = load("Dragon.joblib")
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])

print(model.predict(features))




