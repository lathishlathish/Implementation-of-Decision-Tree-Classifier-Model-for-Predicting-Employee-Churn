# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:


Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:LATHISH KANNA.M
RegisterNumber:  212222230073
```

import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

![image](https://github.com/lathishlathish/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120359170/3960cda3-960c-4284-8f2f-44414d7f462b)
![image](https://github.com/lathishlathish/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120359170/b05fb0ae-4984-4493-90d1-beaf2bd7faf4)
![image](https://github.com/lathishlathish/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120359170/73daa933-1584-4b7f-81ab-bd5a926b6eaf)
![image](https://github.com/lathishlathish/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120359170/0f4b4d17-05b4-4f41-8da1-8c8ab5a0a492)
![image](https://github.com/lathishlathish/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120359170/bde3dd8b-b6f6-49c6-a2c6-0e8de4d91017)
![image](https://github.com/lathishlathish/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120359170/5eeb15ed-eecb-448d-8ca0-bf4839197d22)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
