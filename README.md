# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### step-1: start.
### step-2: Import chardet.
### step-3: Read the dataset.
### step-4: Import SVC from sklearn.
### step-5: Fit the data in the model and run the algorithm.
### step-6: stop.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S JAIGANESH
RegisterNumber:  212222240037

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

*/
```

## Output:

#### data.head()
![Screenshot 2023-06-03 175546](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/e6ffe7d7-9626-4670-a59c-aade7210a7e6)

#### data.tail()
![image](https://github.com/Jaiganesh235/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118657189/4971b395-f542-431d-8dbc-f6be8aeb4a8c)


#### data.info()
![Screenshot 2023-06-03 175721](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/4311d78e-102f-4d5d-817f-7b2615c69165)

#### data.isnull().sum()
![Screenshot 2023-06-03 175739](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/b3b9a6ec-aa9f-42c6-934f-6bde27e20bce)

#### Y_prediction value
![Screenshot 2023-06-03 175808](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/3c2c208a-bf0e-4e58-91ba-039fa57aa02c)

#### Accuracy value
![Screenshot 2023-06-03 175813](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/b00cb652-c51a-496c-a40d-45e3ccbef92d)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
