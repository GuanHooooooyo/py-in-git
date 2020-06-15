from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
import pandas as pd

data = pd.read_csv('diamonds_nan.csv')
#特徵選取
cols_to_use = ['carat','table','depth','x','y','z']
X = data[cols_to_use]
y = data.price
my_pipeline = Pipeline(steps=[('preprocessor',SimpleImputer()),
                              ('model',RandomForestRegressor(n_estimators=50,random_state=0))])

scores = cross_validate(my_pipeline,X,y,cv=5,scoring='neg_mean_absolute_error')
print("MAE scores:\n", scores)
