from __future__ import division
from Travel_Data_Processing import x_test, x_train, y_test, y_train, x, y

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time

#Training trees of different max_depths
max_depth = range(2, 50)
train_err = [0] * len(max_depth)
test_err = [0] * len(max_depth)

for i, d in enumerate(max_depth):
    print 'learning a decision tree with max_depth=' + str(d)
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf = clf.fit(x_train, y_train)

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Decision Trees: Performance x Max Depth')
plt.plot(max_depth, test_err, '-', label='test error')
plt.plot(max_depth, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Mean Square Error')
plt.draw()

print 'here'
# Training trees of different training set sizes (fixed max_depth=8)
train_size = len(x_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
MAX_DEPTH = 35
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
    clf = clf.fit(x_train[:o], y_train[:o])

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Decision Trees: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()

t0 = time.time()
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(x_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(x_test))*100
print("The test works with " + str(accuracy) + "% accuracy")
print(str(time.time() - t0) + " seconds wall time.")
