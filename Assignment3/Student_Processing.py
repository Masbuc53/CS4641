#import statements
import pandas as pd
import numpy as np

#import data
dataTravel = pd.read_csv('studentsPCA.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_PCA = dataTravel[nonTarget]
y_PCA = dataTravel['yes']

#import data
dataTravel = pd.read_csv('studentsICA.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_ICA = dataTravel[nonTarget]
y_ICA = dataTravel['yes']

#import data
dataTravel = pd.read_csv('studentsRP.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_RP = dataTravel[nonTarget]
y_RP = dataTravel['yes']

#import data
dataTravel = pd.read_csv('studentGreedyCfs.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_CFS = dataTravel[nonTarget]
y_CFS = dataTravel['yes']
