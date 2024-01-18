import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter

def DiabetePredictor():
    data = pd.read_csv('Diabetespredictor.csv')

    print("Columns of Dataset")
    print(data.columns)

    print("First 5 records of dataset")
    print(data.head())

    print("Dimension of Diabetes data: {}".format(data.shape))

    X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns!='Outcome'], data['Outcome'], stratify=data['Outcome'],random_state=66)

    logreg = LogisticRegression().fit(X_train,y_train)

    print("Training set accuracy:{:.3f}".format(logreg.score(X_train,y_train)))

    print("Test set accuracy:{:.3f}".format(logreg.score(X_test,y_test)))

    logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)

    print("Training set accuracy:{:.3f}".format(logreg001.score(X_train,y_train)))
    print("Test set accuracy:{:.3f}".format(logreg001.score(X_test,y_test)))

def main():
    simplefilter(action='ignore',category=FutureWarning)
    print("--------------Marvellous Infosystems by Piyush Khainar-------------- ")

    print("---Diabete Predictor Using Logistic Regression-----------")

    DiabetePredictor()

if __name__ == "__main__":
    main()
