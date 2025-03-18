import pandas as pd
import numpy as np
import sklearn


df = pd.read_csv("test_scores.csv")

df.head()

# ban chan

setting_mapping = {'Urban':2, 'Suburban':1, 'Rural':0}
type_mapping = {'Non-public':1, 'Public':0}
teaching_mapping = {'Standard':0,'Experimental':1}
lunch_mapping = {'Does not qualify': 0, 'Qualifies for reduced/free lunch': 1}


df['school_setting'] = df['school_setting'].map(setting_mapping)
df['school_type'] = df['school_type'].map(type_mapping)
df['teaching_method'] = df['teaching_method'].map(teaching_mapping)
df['lunch'] = df['lunch'].map(lunch_mapping)

# X = df[['school_setting','school_type','teaching_method','n_student','lunch','pretest']]
X = df[['school_setting','school_type','teaching_method','n_student','lunch', 'pretest']]
Y = df[["posttest"]]

print(X)


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.2, random_state=42)

Model = sklearn.linear_model.LinearRegression()

Model.fit(X_train,Y_train)
Y_pred = Model.predict(X_test)

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
r2 = sklearn.metrics.r2_score(Y_test,Y_pred)

print(Model.score(X_test, Y_test))
print(mse, r2)

