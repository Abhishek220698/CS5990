#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
df_train = pd.read_csv('weather_training.csv')
df_train.dropna(inplace=True)

#reading the test data
df_test = pd.read_csv('weather_test.csv')
df_test.dropna(inplace=True)
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')

# seperate the independent and target variable on training data
X_training = df_train.drop(columns=['Formatted Date','Temperature (C)'],axis=1)
y_training = df_train['Temperature (C)']

# seperate the independent and target variable on testing data
X_test = df_test.drop(columns=['Formatted Date','Temperature (C)'],axis=1)
y_test = df_test['Temperature (C)']

# loop over the hyperparameter values (k, p, and w) of KNN
#--> add your Python code here
for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            #--> add your Python code here

            #fitting the knn to the data
            # clf = KNeighborsRegressor(n_neighbors=k_values, p=p_values, weights=w_values)
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code herey_test
            
            y_pred = clf.predict(X_test)
            # y_pred = clf.predict(zip(X_test, y_test))

            difference = 100*((y_pred - y_test)/y_test)

            # print("RMS: %r " % np.sqrt(np.mean((y_pred - y_test) ** 2)))

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here
            print(f'Highest KNN accuracy so far: {clf.score(X_test,y_test)}, Parameters: k={k}, p={p}, w={w}')

