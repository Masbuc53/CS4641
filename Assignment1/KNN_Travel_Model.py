from __future__ import division
from Travel_Data_Processing import x_test, x_train, y_test, y_train, x, y

from sklearn import neighbors
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


#Training kNNs of different Ks
ks = range(2, 7)
train_err = [0] * len(ks)
test_err = [0] * len(ks)

for i, k in enumerate(ks):
    print 'learning a kNN classifier with k=' + str(k)
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf = clf.fit(x_train, y_train)

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('KNNClassifier: Performance x K')
plt.plot(ks, test_err, '-', label='test error')
plt.plot(ks, train_err, '-', label='train error')
plt.legend()
plt.xlabel('K')
plt.ylabel('Mean Square Error')
plt.draw()


#Training kNNs of different training set sizes (fixed n_neighbors=5)
train_size = len(x_train)
offsets = range(int(0.1 * train_size), train_size, int(0.1 * train_size))
N_NEIGHBORS = 5
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a kNN classifier with training_set_size=' + str(o)
    clf = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    clf = clf.fit(x_train[:o], y_train[:o])

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('KNN CLassifier: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()

t0 = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(x_test))*100
print("The test works with " + str(accuracy) + "% accuracy")
print(str(time.time() - t0) + " seconds wall time.")
