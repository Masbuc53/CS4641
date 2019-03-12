from Student_Data_Processing import x_test, x_train, y_test, y_train, x, y

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state= 20)

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

Y_prediction = clf.predict(x_test)
accuracy = accuracy_score(y_test, Y_prediction)*100
print("The test works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(y_test, Y_prediction, average = "weighted")*100
loss = log_loss(y_test, Y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, x, y, title = "Learning Curve: Neural Network")
plt.draw()

params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
          'learning_rate_init': 0.2},
         {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
          'nesterovs_momentum': False, 'learning_rate_init': 0.2},
         {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
          'nesterovs_momentum': True, 'learning_rate_init': 0.2},
         {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
          'learning_rate_init': 0.2},
         {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
          'nesterovs_momentum': True, 'learning_rate_init': 0.2},
         {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
          'nesterovs_momentum': False, 'learning_rate_init': 0.2},
         {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
         "constant with Nesterov's momentum",
         "inv-scaling learning-rate", "inv-scaling with momentum",
         "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
            {'c': 'green', 'linestyle': '-'},
            {'c': 'blue', 'linestyle': '-'},
            {'c': 'red', 'linestyle': '--'},
            {'c': 'green', 'linestyle': '--'},
            {'c': 'blue', 'linestyle': '--'},
            {'c': 'black', 'linestyle': '-'}]


def plot_on_dataset(x, y, name):
   # for each dataset, plot learning for each learning strategy
   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.set_title(name)
   X = MinMaxScaler().fit_transform(x)
   mlps = []
   for label, param in zip(labels, params):
       max_iter = 150
       print("training: %s" % label)
       mlp = MLPClassifier(verbose=0, random_state=0,
                           max_iter=max_iter, **param)
       mlp.fit(x, y)
       mlps.append(mlp)
       print("Training set score: %f" % mlp.score(x, y))
       print("Training set loss: %f" % mlp.loss_)
   for mlp, label, args in zip(mlps, labels, plot_args):
           ax.plot(mlp.loss_curve_, label=label, **args)


plot_on_dataset(x, y, "Neural Nets")

plt.show()
