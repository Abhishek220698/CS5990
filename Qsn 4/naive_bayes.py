#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#reading the training data
df_train = pd.read_csv('weather_training.csv')
df_train.dropna(inplace=True)

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
kbins = KBinsDiscretizer(n_bins=11, encode='ordinal', strategy='uniform')
df_train_trans = kbins.fit_transform(df_train.iloc[:,1:])
df_train_trans = DataFrame(df_train_trans)

# seperate the independent and target variable on training data
X_training = df_train_trans.iloc[:,:-1]
y_training = df_train_trans.iloc[:,-1]

#reading the test data
df_test = pd.read_csv('weather_test.csv')
df_test.dropna(inplace=True)

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
# kbins = KBinsDiscretizer(n_bins=11, encode='ordinal', strategy='uniform')
df_test_trans = kbins.fit_transform(df_test.iloc[:,1:])
df_test_trans = DataFrame(df_test_trans)

# seperate the independent and target variable on testing data
X_test = df_test_trans.iloc[:,:-1]
y_test = df_test_trans.iloc[:,-1]


#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
#--> add your Python code here
y_pred = clf.predict(X_test)

#print(f'Percentage difference is: \n{100*((y_pred - y_test)/y_test)}')

#print the naive_bayes accuracyy
#--> add your Python code here
accuracy = clf.score(X_test,y_test)
print("naive_bayes accuracy: " + str(accuracy))