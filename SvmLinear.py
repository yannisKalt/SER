
import pandas as pd
import numpy as np
from tools_no_librosa import split_dataset
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features = pd.read_csv('Global_Features.csv', index_col=0).values
array1 = features[:, :-1]
label = features[:, -1]
xtrain, xtest, ytrain, ytest = split_dataset(array1, label)
xtr_scaled = scaler.fit_transform(xtrain)
xtst = scaler.transform(xtest)

tuned_parameters = [{'kernel': ['linear'], 'C': np.linspace(0.4, 1, num=100)}]
grid = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
grid.fit(xtr_scaled, ytrain)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
y_predict = grid.predict(xtst)
accfinal=accuracy_score(ytest, y_predict)
print(accfinal)
