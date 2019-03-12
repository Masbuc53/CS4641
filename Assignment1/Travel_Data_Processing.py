#import statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#import data
dataTravel = pd.read_csv('../Datasets/tripadvisor_review.csv')

#remove unnecessary data
dataTravel.drop(['User ID'], axis = 1, inplace = True)

#create bins and then categories based on rating values
dataTravel['Religious Institutions'] = dataTravel['Religious Institutions'].apply(np.floor)
dataTravel['Religious Institutions'] = dataTravel['Religious Institutions'].replace([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], ['Bad', 'Bad', 'Bad', 'Good', 'Good', 'Good'])

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('Religious Institutions')

x = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y = pd.get_dummies(dataTravel['Religious Institutions'], prefix_sep='_', drop_first=True)

x_train, x_test = train_test_split(x, test_size = 0.2)
y_train, y_test = train_test_split(y, test_size = 0.2)
