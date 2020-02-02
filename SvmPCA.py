
import pandas as pd
import numpy as np
from tools_no_librosa import split_dataset
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import figure


scaler = StandardScaler()

features = pd.read_csv('Global_Features.csv', index_col=0).values
array1 = features[:, :-1]
print(array1.shape)
label = features[:, -1]
xtrain, xtest, ytrain, ytest = split_dataset(array1, label)
xtr_scaled = scaler.fit_transform(xtrain)
xtst = scaler.transform(xtest)

#Fit and transform with PCA
pca = PCA(0.95)
pca.fit(xtst)

#Plot PCA variance
ax = figure().gca()
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


print("Number of principle components: %s"
      % pca.n_components_)
xtr_scaled = pca.transform(xtr_scaled)
xtst = pca.transform(xtst)

#Svm calculations
clf = svm.SVC()
clf.fit(xtr_scaled, ytrain)
y_predict = clf.predict(xtst)
accfinal=accuracy_score(ytest, y_predict)
print(accfinal)

#Hyper-parameters tuning
C_range =  np.linspace(1.1, 1.6, 80)
gamma_range = np.linspace(0.01, 0.04, 60)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5)
grid.fit(xtr_scaled, ytrain)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
y_predict = grid.predict(xtst)
accfinal=accuracy_score(ytest, y_predict)
print(accfinal)

plt.show()
