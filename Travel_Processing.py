#import statements
import pandas as pd
import numpy as np

#import data
dataTravel = pd.read_csv('travelPCA.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('Good')

x_PCA = dataTravel[nonTarget]
y_PCA = dataTravel['Good']

#import data
dataTravel = pd.read_csv('travelICA.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('Good')

x_ICA = dataTravel[nonTarget]
y_ICA = dataTravel['Good']

#import data
dataTravel = pd.read_csv('travelRP.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('Good')

x_RP = dataTravel[nonTarget]
y_RP = dataTravel['Good']

#import data
dataTravel = pd.read_csv('travelGreedyCfs.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('Good')

x_CFS = dataTravel[nonTarget]
y_CFS = dataTravel['Good']
