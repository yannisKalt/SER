
import sklearn
import pandas as pd
import numpy as np
from tools_no_librosa import split_dataset
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt


scaler = StandardScaler()

features = pd.read_csv('Global_Features.csv', index_col=0).values
array1 = features[:, :-1]
label = features[:, -1]
xtrain, xtest, ytrain, ytest = split_dataset(array1, label)
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

clf = svm.SVC()
clf.fit(xtrain, ytrain)
y_predict = clf.predict(xtest)
accfinal=accuracy_score(ytest, y_predict)
print(accfinal)

#Hyper-parameters tuning
C_range = np.logspace(-2, 10, 40)
gamma_range = np.logspace(-9, 3, 40)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5)
grid.fit(xtrain, ytrain)
classifier = grid
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
y_predict = grid.predict(xtest)
accfinal=accuracy_score(ytest, y_predict)
print(accfinal)

# Plot confusion matrix
titles_options = [("SVM RBF Confusion Matrix", None),
                  ("SVM RBF Normalized Confusion Matrix", 'true')]
for title, normalize in titles_options:
    disp = sklearn.metrics.plot_confusion_matrix(classifier, xtest, ytest,
                                 cmap=plt.cm.magma_r,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

#Print confusion matrix metrics and show the confusion matrix plot
print(metrics.classification_report(ytest, y_predict, digits=4))
plt.show()