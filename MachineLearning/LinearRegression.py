import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


df = pd.read_csv("test_scores.csv")

# df.head()


setting_mapping = {'Urban':2, 'Suburban':1, 'Rural':0}
type_mapping = {'Non-public':1, 'Public':0}
teaching_mapping = {'Standard':0,'Experimental':1}
lunch_mapping = {'Does not qualify': 0, 'Qualifies for reduced/free lunch': 1}

SchoolMeanGroup = df.groupby(['school'])['pretest'].mean()
ClassMeanGroup = df.groupby(['classroom'])['pretest'].mean()
# SchoolMeanGroup['ANKYI']
# ClassMeanGroup


df['school'] = df['school'].map(SchoolMeanGroup)
df['classroom'] = df['classroom'].map(ClassMeanGroup)
df['school_setting'] = df['school_setting'].map(setting_mapping)
df['school_type'] = df['school_type'].map(type_mapping)
df['teaching_method'] = df['teaching_method'].map(teaching_mapping)
df['lunch'] = df['lunch'].map(lunch_mapping)


# X = df[['school_setting','school_type','teaching_method','n_student','lunch','pretest']]
X = df[['school','classroom','school_setting','school_type','teaching_method','n_student','lunch']]
Y = df[["posttest"]]

print(X)

# poly = sklearn.preprocessing.PolynomialFeatures(degree=2)
# X = poly.fit_transform(X)
#
#
# stdscale = sklearn.preprocessing.StandardScaler()
#
# X_train = stdscale.fit_transform(X_train)
# X_test = stdscale.transform(X_test)
#
Model = sklearn.linear_model.LinearRegression()


kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

score = sklearn.model_selection.cross_val_score(Model,X,Y,cv=kfold)
print(np.mean(score))

Model = sklearn.linear_model.Ridge(alpha=0.1)


kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

score = sklearn.model_selection.cross_val_score(Model,X,Y,cv=kfold)
print(np.mean(score))


# Model.fit(X_train,Y_train)
# Y_pred = Model.predict(X_test)
#
# mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
# r2 = sklearn.metrics.r2_score(Y_test,Y_pred)
#
# print(Model.score(X_test, Y_test))
# print(mse, r2)
X = df[['school','classroom','school_setting','school_type','teaching_method','n_student','lunch']]
Y = df[["posttest"]]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.2, random_state=42)

regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=10,random_state=42, oob_score=True)
regressor.fit(X_train,Y_train)

print("oob score: ", regressor.oob_score_ )

predictions = regressor.predict(X_test)

print("MSE= ", sklearn.metrics.mean_squared_error(Y_test,predictions), " R-Squared= ", sklearn.metrics.r2_score(Y_test,predictions))

# treeplot = regressor.estimators_[1]
# plt.figure(figsize = (10,5))
# sklearn.tree.plot_tree(treeplot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
# plt.title("Random Forest's Decision Tree")
# plt.show()

regressor = sklearn.ensemble.RandomForestRegressor(random_state=42)
hyperparameters = {'max_depth':[2,3,5,10,20,30],
                   'n_estimators':[10,50,100,200,300],
                   'min_samples_split': [2,5,7],
                   'min_samples_leaf':[1,2,4],
                   'max_features': ['auto','sqrt','log2']
                   }
grid_search = sklearn.model_selection.GridSearchCV(estimator=regressor, param_grid=hyperparameters, cv=5, scoring='r2',n_jobs=-1)
grid_search.fit(X_train,Y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Best parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
# Best score: 0.9479861982663854


























