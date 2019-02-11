from __future__ import division
from Student_Data_Processing import x_test, x_train, y_test, y_train, x, y

from sklearn import ensemble, tree
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

#Training trees of different n_estimators
max_n_estimators = range(5, 40, 5)
train_err = [0] * len(max_n_estimators)
test_err = [0] * len(max_n_estimators)

for i, e in enumerate(max_n_estimators):
    print 'learning a decision tree with n_estimators=' + str(e) + ' (fixed max_depth=10)'
    t = tree.DecisionTreeClassifier(max_depth=10)
    clf = ensemble.AdaBoostClassifier(base_estimator=t, n_estimators=e)
    clf = clf.fit(x_train, y_train)

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Boosted Decision Trees: Performance x Num Estimators')
plt.plot(max_n_estimators, test_err, '-', label='test error')
plt.plot(max_n_estimators, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.draw()


#Training trees of different training set sizes (fixed max_depth=10, n_estimators=10)
train_size = len(x_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
MAX_DEPTH = 10
N_ESTIMATORS = 10
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o) + ' (fixed max_depth=10, n_estimators=10)'
    t = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
    clf = ensemble.AdaBoostClassifier(base_estimator=t, n_estimators=N_ESTIMATORS)
    clf = clf.fit(x_train[:o], y_train[:o])

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Boosted Decision Trees: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()

t0 = time.time()
t = tree.DecisionTreeClassifier(max_depth=10)
clf = ensemble.AdaBoostClassifier(base_estimator=t, n_estimators=15)
clf.fit(x_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(x_test))*100
print("The test works with " + str(accuracy) + "% accuracy")
print(str(time.time() - t0) + " seconds wall time.")
