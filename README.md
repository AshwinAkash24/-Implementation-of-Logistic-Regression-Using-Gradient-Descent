# EX-06:Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### STEP-1:
First, the data is loaded from a CSV file. Two columns that are not needed (sl_no and salary) are removed. Then, all the text columns like gender and education type are changed into numbers so the model can use them.
#### Step-2:
Next, the data is split into input (X) and output (Y). X has all the features, and Y has the placement result (placed or not).
#### Step-3:
Then, some random values are set for the model to start with. A sigmoid function is made, which turns numbers into probabilities between 0 and 1.
#### Step-4:
The model is trained using a method called gradient descent. It keeps changing the values to make the predictions better by checking the error and fixing it again and again.
#### Step-5:
After training, the model makes predictions. If the result is 0.5 or more, it's marked as placed (1). If it's less, it's not placed (0). The predictions are checked with the real results to see how accurate the model is.
#### Step-6:
In the end, you try to predict for a new student. But the input needs to be in the right shape (2D) for it to work properly.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Ashwin Akash M
RegisterNumber:  212223230024
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('Placement_Data.csv')
df
df=df.drop(['sl_no','salary'],axis=1)
df['gender']=df['gender'].astype('category')
df['ssc_b']=df['ssc_b'].astype('category')
df['hsc_b']=df['hsc_b'].astype('category')
df['hsc_s']=df['hsc_s'].astype('category')
df['degree_t']=df['degree_t'].astype('category')
df['workex']=df['workex'].astype('category')
df['specialisation']=df['specialisation'].astype('category')
df['status']=df['status'].astype('category')
df.dtypes
df['gender']=df['gender'].cat.codes
df['ssc_b']=df['ssc_b'].cat.codes
df['hsc_b']=df['hsc_b'].cat.codes
df['hsc_s']=df['hsc_s'].cat.codes
df['degree_t']=df['degree_t'].cat.codes
df['workex']=df['workex'].cat.codes
df['specialisation']=df['specialisation'].cat.codes
df['status']=df['status'].cat.codes
df
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)
acc=np.mean(y_pred.flatten()==y)
print("Accuracy=",acc)
print(y_pred)
print(Y)
xn=np.array([0,87,0,95,0,2,78,2,0,0,1,0])
ypr=predict(theta,xn)
ypr
```

## Output:
![image](https://github.com/user-attachments/assets/a0509410-7bd8-44fd-b4ff-a85a9b8ac164)<br>
![image](https://github.com/user-attachments/assets/247702f0-6c95-4d25-a32c-93c8df3720ba)<br>
![image](https://github.com/user-attachments/assets/dca1c6a8-b125-413b-bf7b-8992aef38a95)<br>
![image](https://github.com/user-attachments/assets/be5ab203-5325-4e17-afc4-245a77ff4739)<br>
![image](https://github.com/user-attachments/assets/f9de73ce-38d7-4b6c-8e95-6ead35f9dbb1)<br>
![image](https://github.com/user-attachments/assets/22251014-b3b4-4612-bca6-e60cf59e2127)<br>
![image](https://github.com/user-attachments/assets/2631357c-fcc3-40e8-b3e4-539b4f6c38f1)<br>
![image](https://github.com/user-attachments/assets/cafba12d-1c8e-4dc3-b4e5-2915b9a4621e)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

