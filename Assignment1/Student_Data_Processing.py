#import statements
import pandas as pd
from sklearn.model_selection import train_test_split

#import data
dataMath = pd.read_csv('../Datasets/student/student-mat.csv')

#create masks for target and non-target data
nonTarget = dataMath.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('internet')

#create my training and testing datasets
x = pd.get_dummies(dataMath[nonTarget], prefix_sep='_', drop_first=True)
y = pd.get_dummies(dataMath['internet'], prefix_sep='_', drop_first=True)

x_train, x_test = train_test_split(x, test_size = 0.2)
y_train, y_test = train_test_split(y, test_size = 0.2)
