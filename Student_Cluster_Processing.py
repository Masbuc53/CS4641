#import statements
import pandas as pd
import numpy as np

#import data
dataTravel = pd.read_csv('studentsPCACluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_PCA_EM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_PCA_EM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)

#import data
dataTravel = pd.read_csv('studentsICACluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_ICA_EM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_ICA_EM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)

#import data
dataTravel = pd.read_csv('studentsRPCluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_RP_EM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_RP_EM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)

#import data
dataTravel = pd.read_csv('studentsGreedyCfsCluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_CFS_EM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_CFS_EM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)

#import data
dataTravel = pd.read_csv('studentsPCAkmCluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_PCA_KM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_PCA_KM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)

#import data
dataTravel = pd.read_csv('studentsICAkmCluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_ICA_KM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_ICA_KM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)

#import data
dataTravel = pd.read_csv('studentsRPkmCluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_RP_KM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_RP_KM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)

#import data
dataTravel = pd.read_csv('studentsGreedyCfsKmCluster.csv')

#create my training and testing datasets
nonTarget = dataTravel.keys()
nonTarget = nonTarget.tolist()
nonTarget.remove('yes')

x_CFS_KM = pd.get_dummies(dataTravel[nonTarget], prefix_sep='_', drop_first=True)
y_CFS_KM = pd.get_dummies(dataTravel['yes'], prefix_sep='_', drop_first=True)
