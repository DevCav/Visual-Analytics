## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=100, shuffle=True)


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)
y_train_pred = regr.predict(x_train)


# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX


testing_accuracy = accuracy_score(y_test.round(), y_pred.round())
training_accuracy = accuracy_score(y_train.round(), y_train_pred.round())

# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

clf_y_pred = clf.predict(x_test)
clf_y_train_pred = clf.predict(x_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

clf_testing_accuracy = accuracy_score(y_test.round(), clf_y_pred.round())
clf_training_accuracy = accuracy_score(y_train.round(), clf_y_train_pred.round())

# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX

important_features_dict = {}
for x,i in enumerate(clf.feature_importances_):
    important_features_dict[x]=i


important_features_list = sorted(important_features_dict,
                                 key=important_features_dict.get,
                                 reverse=True)


# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX

max_depth = [10, 80, None]

n_estimators = [200, 1000, 2000]

# Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth}

CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 10)
CV_rfc.fit(x_train, y_train)

print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX

normalized_x_train = normalize(x_train)
normalized_x_test = normalize(x_test)

svc = SVC()
svc.fit(normalized_x_train, y_train) 

svc_y_pred = svc.predict(normalized_x_test)
svc_y_train_pred = svc.predict(normalized_x_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

svc_testing_accuracy = accuracy_score(y_test.round(), svc_y_pred.round())
svc_training_accuracy = accuracy_score(y_train.round(), svc_y_train_pred.round())



# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX

 C = [0.001, 1, 10]
 kernel = ['rbf', 'linear']
 
 param_grid_svc = {'C': C,
               'kernel': kernel}
 
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=10)
grid_search.fit(normalized_x_train, y_train)

tuned_svc = SVC(kernel = 'rbf', C = .001)
tuned_svc.fit(normalized_x_train, y_train) 
tuned_svc_y_pred = tuned_svc.predict(normalized_x_test)
tuned_svc_testing_accuracy = accuracy_score(y_test.round(), tuned_svc_y_pred.round())
# XXX
# TODO: Calculate the mean training score, mean testing score and mean fit time for the 
# best combination of hyperparameter values that you obtained in Q3.2. The GridSearchCV 
# class holds a  ‘cv_results_’ dictionary that should help you report these metrics easily.
# XXX

# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX
 
pca = PCA(n_components=10, svd_solver='full')
pca.fit(x_train)

print(list(pca.explained_variance_ratio_)) 
print(list(pca.singular_values_))

