# -*- coding: utf-8 -*-
"""
Spyder Editor

First time trying to make a model for Kaggle
"""

# Import relevant Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Import tuple for converting titles 
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

# Helper function to save as CSV
def submitResult(testSet, yPredict):
    yPredict = yPredict.astype(int)
    submission = pd.concat([testSet['PassengerId'], yPredict], axis=1, 
                           join_axes=[testSet.index])
    submission.set_index('PassengerId')
    submission.rename(columns={0:'Survived'}, inplace='True')

    submission.to_csv('prediction.csv', index=False)

train = pd.read_csv('train.csv') 
test = pd.read_csv('test.csv')

# Merge the data to one dataframe for cleaning 
data = pd.concat([train, test], sort='false')

# Fix age missing data
data.loc[data['Sex'] == 'male' , 'Age'] = data.loc[data['Sex'] == 'male' , 'Age'].fillna(\
        data.loc[data['Sex'] == 'male' , 'Age'].median())
data.loc[data['Sex'] == 'female' , 'Age'] = data.loc[data['Sex'] == 'female' , 'Age'].fillna(\
        data.loc[data['Sex'] == 'female' , 'Age'].median())

# Replace missing fare data with the median
# Fill the training Set
data.loc[data.index[:891], 'Fare'] = data.loc[data.index[:891], 'Fare'].fillna(\
        train['Fare'].median())
# Fill the test set
data.loc[data.index[891:], 'Fare'] = data.loc[data.index[891:], 'Fare'].fillna(\
        test['Fare'].median())
# Categorize the age and Fare into bins
data['CatAge'] = pd.qcut(data['Age'], q=5, labels=False)
data['CatFare'] = pd.qcut(data['Fare'], q=4, labels=False)


# Create a new feature called Status
data['Status'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
data['Status'] = data['Status'].fillna('None')
# Replace redundant titles with proper ones
data['Status'] = data['Status'].map(lambda status:Title_Dictionary[status])
      
# Create a new feature to replace old cabin with boolean
data['Has Cabin'] = ~data['Cabin'].isnull()
# Replace missing cabin data with Z
data['Cabin'] = data['Cabin'].fillna('Z')
# Creates a new feature Cabin out of the first letter from Cabin
data['Deck'] = data['Cabin'].map(lambda cabin:cabin[0].strip())

# Create a new Feature Family Size
data['Family Size'] = data['Parch'] + data['SibSp']
# Travelling alone feature
data['Is Alone'] = 0 
data.loc[data['Family Size'] == 1, 'Is Alone'] = 1

# Replace missing embarked values with U
data["Embarked"] = data["Embarked"].fillna("U")
# Encode the Embarked
data.loc[data['Embarked'] == 'C' , 'Ch'] = 1
data.loc[data['Embarked'] == 'Q' , 'Qu'] = 1
data.loc[data['Embarked'] == 'S' , 'So'] = 1
data.loc[data['Embarked'] == 'U' , 'Un'] = 1
# Encode the Sex class
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Sex'] == 'female', 'Sex'] = 0
# Encode P class into dummy variables
data.loc[data['Pclass'] == 1 , 'Class 1'] = 1 
data.loc[data['Pclass'] == 2 , 'Class 2'] = 1
data.loc[data['Pclass'] == 3 , 'Class 3'] = 1
# Encode the Title Class
dummy = pd.get_dummies(data['Status'], drop_first=True)
data = pd.concat([data, dummy], axis=1)
# Encode the hasCabin Class
dummy = pd.get_dummies(data['Has Cabin'], drop_first=True)
data = pd.concat([data, dummy], axis=1)
data.rename(columns={True:'HasCabin'}, inplace=True)
# Encode the Deck Categories
dummy = pd.get_dummies(data['Deck'], drop_first=True)
#data = pd.concat([data, dummy], sort=False, axis=1)

# Fill the new dummy categories with 0 for NaN cases
data[['Ch','Qu','Un','So','Class 1','Class 2','Class 3']] = \
data[['Ch','Qu','Un','So','Class 1','Class 2','Class 3']].fillna(0)


# Drop the classes not needed anymore
data = data.drop({'Embarked', 'Pclass', 'Cabin','Deck', 
                  'Parch', 'SibSp','Status','Has Cabin',
                  'Age', 'Fare'}, axis=1)

# Resplit into the original training and test test
# Test set had survived = NaN so use that as a filter
trainSet = data.loc[data['Survived'].notna()]
testSet = data.loc[data['Survived'].isna()]

# Setup hyperparamter gridn make predictions on your tes
dep = np.arange(1,9)
param_grid = {'min_samples_leaf': dep}
# Instantiate a decision tree classifier: clf
clf = RandomForestClassifier(n_estimators=30, max_depth=7)
# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)
# List of features I want to use
xTrain = trainSet.drop({'Ticket', 'Name', 'Survived'},axis=1)
yTrain = trainSet['Survived']
xTest = testSet.drop({'Ticket', 'Name', 'Survived'}, axis=1)
clf_cv.fit(xTrain, yTrain)
yPredict = clf_cv.predict(xTest)
submitResult(testSet, pd.DataFrame(yPredict))

print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))

"""
# Splitting training set to training set and Cross Validation(CV)
Xtrain, Xcv, Ytrain, Ycv =  train_test_split(trainSet.drop('Survived', axis=1), trainSet['Survived'], test_size = 0.4)
       
# Split fare and age into quartiles
data['CatAge'] = pd.qcut(data['Age'], q=4, labels=False)
data['CatFare'] = pd.qcut(data['Fare'], q=4, labels=False)
"""