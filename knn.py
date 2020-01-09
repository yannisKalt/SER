import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tools_no_librosa import split_dataset
from sklearn.metrics import confusion_matrix
from image_annotated_heatmap import *

df = pd.read_csv('Global_Features.csv')

#KNN algorithm x,y values
X = np.array(df.drop(['sentiment'], 1))
y = np.array(df['sentiment'])


#x_train,y_train values
X_train, X_test, y_train, y_test = split_dataset(X,y)

#Standart scaler(kanonikopoihsh)
standard_scaler = preprocessing.StandardScaler()
standard_scaler.fit(X_train)
X_test = standard_scaler.transform(X_test)
X_train = standard_scaler.transform(X_train)


#KNN algorithm(fitting k-0nn to the training set)
clf = neighbors.KNeighborsClassifier(n_neighbors=14,weights='distance',algorithm='ball_tree',metric='canberra',leaf_size=66,p=2)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

#Predicting the test set results
y_pred = clf.predict(X_test)

#Model evaluation step
#Making confusion Matrix

labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness']
cm = confusion_matrix(y_test, y_pred, labels = labels)

#con matrix visualisation
fig, ax = plt.subplots()
im, cbar = heatmap(cm, labels, labels, ax = ax, cmap = 'magma_r')
text = annotate_heatmap(im)
plt.title('KNN Confusion Matrix')
fig.tight_layout()
plt.show()
