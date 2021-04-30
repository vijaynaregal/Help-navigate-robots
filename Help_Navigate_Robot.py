#!/usr/bin/env python
# coding: utf-8

# # Project-Title - Help_Naviagte_Robots

# #  Introduction

# We are given IMU(Inertial Measurement Units) data that is collected by a robot while moving on different types of floor surface. We are going to train our machine learing model on the data and test our data and hopefully be able to correctly predict the type of floor surface.

# **A.Importing the Packages**

# In[1]:


# Importing libraries and packages:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# **B.Loading the Data**

# In[2]:


# reading a CSV file directly from Web (or local drive), and store it in a pandas DataFrame:
X_train = pd.read_csv("https://raw.githubusercontent.com/lkampat/project_4661/main/X_train.csv")
Y_train = pd.read_csv("https://raw.githubusercontent.com/lkampat/project_4661/main/y_train.csv")


# **C.Data Exploration**

# In[3]:


X_train


# There are 3810 series of input data. Each series contains 128 sets of measurements,thereby each measurements were taken 128 times during the course of a walk, So total 487680  values of data.

# In[23]:


#check the shape of the DataFrame (rows, columns):
X_train.shape,Y_train.shape


# In[5]:


Y_train


# **D.Plotting the Y_train graph**

# In[6]:


Y_train['surface'].value_counts().reset_index().plot(x='index',y='surface',kind='bar')


# This is a multi-class classification problem. It's supervised and has imbalanced classes. Each measurement has 128 data points taken over time for each sensor. The identifier for each measurement is series_id. Then each measurement is repeated on the same surface multiple times which is identified by group_id. Each group_id is a unique recording session and has only one surface type

# In[7]:


def plot_series(series):
    df_train=X_train[X_train['series_id']==series]
    
    plt.figure(figsize=(30,15))
    for i,col in enumerate(df_train.columns[3:]):
        plt.subplot(3,4,i+1)
        df_train[col].plot(color='red')
        plt.title(col)

plot_series(0)


# Orientation X increases , Orientation Y decreases and Strong correlation between angular velocity Z and angular velocity Y
# 

# **E.Preprocessing of Data**

# we are converting the 4-degree orientation value into 3-degree euler angle value for better understanding. New values: roll, pitch and yaw, and add them to corresponding row.

# **Converting the Quaternions to Roll,Pitch,Yaw**

# In[8]:


#Function to convert quaternion to euler angles
def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (w * y - z * x)
    if sinp > 1.0:
        sinp = 1.0
    if sinp < -1.0:
        sinp = -1.0
    pitch = math.asin(sinp)
    
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


# (x, y, z, w) -> (x,y,z) quaternions to euler angles

# In[9]:


#function to add euler features to each row
def add_euler_features(dataframe):
    x, y, z, w = dataframe["orientation_X"].tolist(), dataframe["orientation_Y"].tolist(), dataframe["orientation_Z"].tolist(), dataframe["orientation_W"].tolist()
    rolls, pitches, yaws = [], [], []
    for i in range(len(x)):
        roll, pitch, yaw = quaternion_to_euler(x[i], y[i], z[i], w[i])
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
    dataframe["roll"] = rolls
    dataframe["pitch"] = pitches
    dataframe["yaw"] = yaws


# In[10]:


import math
add_euler_features(X_train)


# **F. Feature Engineering**

# we are grouping all of our input data by their series_id and took the mean and sum of each attributes within each series and stored the output into a new dataframe. The returned dataframe will be our new sample data for our machine learning model.

# In[11]:


def feature_engineering(dataframe):
    ret_dataframe = pd.DataFrame()
    
    for column in dataframe.columns:
        if column == "row_id" or column == "measurement_number":
            continue
        ret_dataframe[column + "_mean"] = dataframe.groupby(["series_id"])[column].mean()
        ret_dataframe[column + "_sum"] = dataframe.groupby(["series_id"])[column].sum()

    return ret_dataframe


# In[12]:


X_train = feature_engineering(X_train)
X_train


# **G.Transform "surface" data from categorical data to numerical data.**

# Converting all our Surface classification in numerical data in Encoded surface column in Y_train Data. It is required so that the classifier gets discrete integers from 0 to 9 in order to classify properly

# In[13]:


lableDict = {"concrete": 0, "soft_pvc": 1, "wood": 2, "tiled": 3, "fine_concrete": 4, 
             "hard_tiles_large_space": 5, "soft_tiles": 6, "carpet": 7, "hard_tiles": 8}
def encodeSurface(y_dataframe):
    surface = y_dataframe["surface"].tolist()
    encodedSurface = []
    for i in range(len(surface)):
        encodedSurface.append(lableDict[surface[i]])
    y_dataframe["encoded_surface"] = encodedSurface

encodeSurface(Y_train)
Y_train.head()


# **H. Merge the data**

# The "series_id" is the primary key in y_train, and it's also the foreign key to y_train in X_train.
# Its easier for training that each example in X to have a y label therefore we merge the X dataframe and y dataframe accordingly using the primary key series_id.

# In[14]:


X_train['encoded_surface']=Y_train['encoded_surface']
X_train['group_id']=Y_train['group_id']
X_train['surface']=Y_train['surface']

X_train.head()


# **I. Fitting and Predicting the data**

# **Function to Predict ROC_AUC score using LabelBinarizer for multiclass for each classifier**

# In[15]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

def multiclass_roc_auc_score(y_test, y_predict_classifier, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_predict_classifier)
    return roc_auc_score(y_test, y_pred, average=average)


# **Function to fit and Predict the model**

# In[16]:


def my_classifer(classifier,X_train,y_train,X_test,y_test,class_names):
    
    newdict = {}
    #fit the model
    classifier.fit(X_train, y_train)
    
    #Predict the model
    y_predict_label = classifier.predict(X_test)
    
    #performed various performance metric for this model:
    #get accuracy_score for the model
    accuracy = accuracy_score(y_test, y_predict_label)
    print('\n Accuracy:', accuracy)
    
    #get Auc score for the model
    roc_auc_score=multiclass_roc_auc_score(y_test, y_predict_label)
    print("\n AUC Score_LableBinarizer:",roc_auc_score)
    
    #get cross-validation score for the model
    accuracy_list = cross_val_score(classifier, X, Y, cv=10, scoring='accuracy')
    accuracy_cv = accuracy_list.mean()
    print('\n Accuracy_cv-',accuracy_cv)
    
    return y_predict_label
   


# **Feature Matrix and label data**

# In[17]:


## create a python list of feature names that would like to pick from the dataset:
feature_cols=['series_id_mean','angular_velocity_X_mean',
 'angular_velocity_X_sum',
 'angular_velocity_Y_mean',
 'angular_velocity_Y_sum',
 'angular_velocity_Z_mean',
 'angular_velocity_Z_sum',
 'linear_acceleration_X_mean',
 'linear_acceleration_X_sum',
 'linear_acceleration_Y_mean',
 'linear_acceleration_Y_sum',
 'linear_acceleration_Z_mean',
 'linear_acceleration_Z_sum',
 'roll_mean',
 'roll_sum',
 'pitch_mean',
 'pitch_sum',
 'yaw_mean',
 'yaw_sum']

# use the above list to select the features from the  DataFrame
X = X_train[feature_cols] 
# select a Series of labels (the last column) from the DataFrame
Y=X_train['encoded_surface']
Y_target=X_train['surface']


#preprocessing the data
from sklearn import preprocessing
X = preprocessing.scale(X)
X


# **J.Splitting the Dataset:**

# In[18]:


from sklearn.model_selection import train_test_split

# Randomly splitting the original dataset into training set and testing set:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=3)


# **1.K-Nearest Neighbors Classifier**

# In[19]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from statistics import mean

print('\n KNN CLASSIFIER \n')
print("------------------")

class_names=Y_target
k = 10
my_knn = KNeighborsClassifier(n_neighbors=k)
y_predict_label=my_classifer(my_knn,X_train,y_train,X_test,y_test,class_names)

# Build the confusion matrix of our 3-class classification problem
cnf_matrix = confusion_matrix(y_test, y_predict_label)


print('\n Confusion Matrix')

#plot Confusion matric for the given classifier
disp = plot_confusion_matrix(my_knn, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
plt.show()
    
    
#calculating TPR and FPR based on plotted confusion matrix

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR1 = TP/(TP+FN)
print("\n KNN Classifier Tpr:",TPR1)
# Specificity or true negative rate
FPR1 = FP/(TN+FP)
print("\n KNN Classifier Fpr:",FPR1)

# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Overall Accuracy of the class:",ACC)


    


# KNN calculates the distance between the train data and test data by picking the nearest K data points. Since a lot of the data from different class could be very close to each other which is hard for KNN to identify the accurate target for the corresponding features. Therefore, this model did not produce high accuracy compared to other ML models. Implemented confusion matrix to find the sensitivity and specificity for all the classes/labels.
# 

# **2 . Decision tree Classifier**

# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from statistics import mean

print('\n DECISION TREE CLASSIFIER \n')
print("________________________________")

class_names=Y_target
my_dt=DecisionTreeClassifier(random_state=3)
y_predict_label=my_classifer(my_dt,X_train,y_train,X_test,y_test,class_names)

# Build the confusion matrix of our 3-class classification problem
cnf_matrix = confusion_matrix(y_test, y_predict_label)

 
print('\n Decision tree Confusion Matrix')    
    
#plot Confusion matric for the given classifier
disp = plot_confusion_matrix(my_dt, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
plt.show()
    
    
#calculating TPR and TNR based on plotted confusion matrix
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR2 = TP/(TP+FN)
print("\n Decision tree Tpr:",TPR2)
# Specificity or true negative rate
FPR2 = TP/(TN+FP)
print("\n Decision tree Fpr:",FPR2)

# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Overall Accuracy of the class from confusion matrix:",mean(ACC))


    


# Decision tree classifier uses divide and conquer algorithm which splits the training data on feature that has the highest information gain till it is sufficient enough to identify the different classes recursively producing higher accuracy compared to KNN classifier. We implemented confusion matrix to find the sensitivity and specificity for all the classes/labels.
# 

# **3.Random Forest Classifier**

# In[21]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from statistics import mean

print('\n RANDOM FOREST \n')
print("_______________________")

class_names=Y_target
my_RandomForest = RandomForestClassifier(n_estimators = 19, bootstrap = True, random_state=3)
y_predict_label=my_classifer(my_RandomForest,X_train,y_train,X_test,y_test,class_names)


print('\n Random Forest Confusion Matrix \n')
# Build the confusion matrix of our 3-class classification problem

cnf_matrix = confusion_matrix(y_test, y_predict_label)

#plot Confusion matric for the given classifier
disp = plot_confusion_matrix(my_RandomForest, X_test, y_test,display_labels=class_names,cmap=plt.cm.Blues)
plt.show()
    
    
#calculating TPR and TNR based on plotted confusion matrix
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR3 = TP/(TP+FN)
print("\n Random forest Tpr : ",TPR3)
# Specificity or true negative rate
FPR3 = FP/(TN+FP)
print("\n Random forest Fpr:",FPR3)

# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Overall Accuracy of the class from confusion matrix:", mean(ACC))



# Random Forest Classifier contains a random collection of decision tree classifier where it merges all decision tree for better accuracy. The default value we used for amount of decision tree is 100, so more the trees, better the accuracy when compared to the above ML classifiers. We implemented confusion matrix to find the sensitivity and specificity for all the classes/labels.
# 

# **K.Plot Roc Curve of all classifiers**

# In[22]:


# Importing the "pyplot" package of "matplotlib" library of python to generate 
# graphs and plot curves:
import matplotlib.pyplot as plt

# The following line will tell Jupyter Notebook to keep the figures inside the explorer page 
# rather than openng a new figure window:
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn')

 
# plot roc curves
plt.plot(FPR1, TPR1, linestyle='--',color='orange', label='KNN ')
plt.plot(FPR2, TPR2, linestyle='--',color='green', label='Decision Tree')
plt.plot(FPR3, TPR3, linestyle='--',color='Yellow', label='Random Forest')
plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.xlim([-0.005, 1.0])
plt.ylim([0.0, 1.0])

plt.legend(loc='best')
#plt.savefig('ROC',dpi=300)
plt.show();


# **From the graph above, it depicts that the "Random forest" curve is almost nearer to the upper corner when compared with KNN and decsion tree, thus resulting as the best model for this particular project with the accuracy of 0.78**

# In[ ]:




