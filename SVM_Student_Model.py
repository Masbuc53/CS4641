from __future__ import division
from Student_Data_Processing import x_test, x_train, y_test, y_train, x, y

from sklearn import svm
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

#Training SVMs of different degrees
degrees = range(6)
train_err = [0] * len(degrees)
test_err = [0] * len(degrees)

for i, d in enumerate(degrees):
    print 'learning an SVM with degree=' + str(d)
    clf = svm.SVC(kernel='poly', degree=d)
    clf = clf.fit(x_train, y_train)

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('SVM: Performance x Degree')
plt.plot(degrees, test_err, '-', label='test error')
plt.plot(degrees, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Mean Square Error')
plt.draw()


#Training SVMs of different training set sizes (fixed degree=3)
train_size = len(x_train)
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))
MAX_DEPTH = 35
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning an SVM with training_set_size=' + str(o)
    clf = svm.SVC()
    clf = clf.fit(x_train[:o], y_train[:o])

    train_err[i] = mean_squared_error(y_train, clf.predict(x_train))
    test_err[i] = mean_squared_error(y_test, clf.predict(x_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('SVMs: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()

t0 = time.time()
clf = svm.SVC(degree=1)
clf.fit(x_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(x_test))*100
print("The test works with " + str(accuracy) + "% accuracy")
print(str(time.time() - t0) + " seconds wall time.")
