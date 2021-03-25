from libextract import *
from randomf import *
from dataclean import *

dataset = pd.read_csv("/Users/Marzieh/Desktop/oop practice/Position_Salaries.csv")
x_columns='Level'
y_columns='Salary'
test_size=0.2
random_state=0
kind='LR'   ## Linear Regression=LR, Decision Tree=DT, Random Forest=RF, Polynomial Regression=PR, SVR=SVR
degree=4
n_estimators=10
kernel='rbf'
dataclean=dataclean(dataset=dataset,x_columns=x_columns,y_columns=y_columns,test_size=test_size,random_state=random_state)
traintest=dataclean.traintest()
regressor=regression(X_train=traintest['X_train'],X_test=traintest['X_test'],y_train=traintest['y_train'],y_test=traintest['y_test'],n_estimators=n_estimators,random_state=random_state,kind=kind,kernel=kernel)
predict=regressor.predict()
print(predict)