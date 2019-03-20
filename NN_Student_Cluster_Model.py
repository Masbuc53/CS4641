from __future__ import division
from Student_Cluster_Processing import x_PCA_EM, y_PCA_EM, x_ICA_EM, y_ICA_EM, x_RP_EM, y_RP_EM, x_CFS_EM, y_CFS_EM, x_PCA_KM, y_PCA_KM, x_ICA_KM, y_ICA_KM, x_RP_KM, y_RP_KM, x_CFS_KM, y_CFS_KM

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_PCA_EM, y_PCA_EM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_PCA_EM, y_PCA_EM, title = "Learning Curve: Neural Network PCA EM")
plt.draw()

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_ICA_EM, y_ICA_EM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_ICA_EM, y_ICA_EM, title = "Learning Curve: Neural Network ICA EM")
plt.draw()

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_RP_EM, y_RP_EM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_RP_EM, y_RP_EM, title = "Learning Curve: Neural Network RP EM")
plt.draw()

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_CFS_EM, y_CFS_EM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_CFS_EM, y_CFS_EM, title = "Learning Curve: Neural Network CFS EM")
plt.draw()

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_PCA_KM, y_PCA_KM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_PCA_KM, y_PCA_KM, title = "Learning Curve: Neural Network PCA K-Means")
plt.draw()

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_ICA_KM, y_ICA_KM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_ICA_KM, y_ICA_KM, title = "Learning Curve: Neural Network ICA K-Means")
plt.draw()

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_RP_KM, y_RP_KM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_RP_KM, y_RP_KM, title = "Learning Curve: Neural Network RP K-Means")
plt.draw()

#start timer
t0= time.time()

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
x_train, x_test, y_train, y_test = train_test_split(x_CFS_KM, y_CFS_KM, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

#predict the training and test accuracy
train_prediction = clf.predict(x_train)
trainaccuracy = accuracy_score(train_prediction, y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, y_prediction, average = "weighted")*100
loss = log_loss(y_test, y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x_CFS_KM, y_CFS_KM, title = "Learning Curve: Neural Network CFS K-Means")
plt.show()
