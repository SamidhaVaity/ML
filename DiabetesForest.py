import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter

def DiabetePredictor():
    data = pd.read_csv('Diabetespredictor.csv')

    print("Columns of Dataset")
    print(data.columns)

    print("First 5 records of dataset")
    print(data.head())

    print("Dimension of Diabetes data: {}".format(data.shape))

    X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns!='Outcome'], data['Outcome'], stratify=data['Outcome'],random_state=66)

    rf = RandomForestClassifier(n_estimators=100,random_state=0)
    rf.fit(X_train,y_train)

    print("Training set accuracy:{:.3f}".format(rf.score(X_train,y_train)))

    print("Test set accuracy:{:.3f}".format(rf.score(X_test,y_test)))

    rf1 = RandomForestClassifier(max_depth=3, n_estimators=100,random_state=0)
    rf1.fit(X_train,y_train)

    print("Training set accuracy:{:.3f}".format(rf1.score(X_train,y_train)))

    print("Test set accuracy:{:.3f}".format(rf1.score(X_test,y_test)))

    def plot_feature_importances_diabetes(model):
            plt.figure(figsize=(8,6))
            n_features = 8
            plt.barh(range(n_features),model.feature_importances_,align='center')
            diabetes_features = [x for i,x in enumerate(data.columns) if i!=8]
            plt.yticks(np.arange(n_features), diabetes_features)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.ylim(-1, n_features)
            plt.show()
    plot_feature_importances_diabetes(rf)
def main():
    simplefilter(action='ignore',category=FutureWarning)
    print("--------------Marvellous Infosystems by Piyush Khainar-------------- ")

    print("---Diabete Predictor Using Random Forest-----------")

    DiabetePredictor()

if __name__ == "__main__":
    main()
