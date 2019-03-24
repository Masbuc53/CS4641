Stanley "Mason" Buchanan
CS 4641 Assignment 3

Codebase: https://github.com/Masbuc53/CS4641/tree/master/Assignment3

Dependencies:

WEKA (latest version)
Python 3
Student Filters (http://weka.sourceforge.net/packageMetaData/StudentFilters/index.html)

The analysis was almost entirely done in WEKA, so be sure to have that downloaded. 
I utilized the WEKA GUI and, through the explorer, manipulated my data, built the clusters, and applied dimensionality 
reduction.

To start, start up WEKA and use the package manager to install the Student Filters. 
From there, begin the analyes in the WEKA explorer by importing the students.csv and travel.csv files and applying a 
NumericaltoNominal filter to the last feature. Once this is done you can perform the appropriate feature selection and 
clustering through the filters in the preprocessing tab and the clustering tab, respectively. The CFS feature selection
algorithm can be found in the feature select tab. After each analysis, download the relevant datafile as a csv.

Once the analyses are done, run the Travel_Processing.py and Student_Processing.py files in the command line by traversing
to the appropriate folder and using the command python ./FILENAME. After that, run the NN_Travel_Model.py and NN_Student_Model.py
files using the same command. Then run the Student_Cluster_Processing.py file and then the NN_Student_Cluster_Model.py file 
using the same command.
