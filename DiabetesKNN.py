import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def DiabetePredictor():
    data = pd.read_csv('Diabetespredictor.csv')

    print("Columns of Dataset")
    print(data.columns)

    print("First 5 records of dataset")
    print(data.head())

    print("Dimension of Diabete data: {}".format(data.shape))

    X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns!='Outcome'], data['Outcome'], stratify=data['Outcome'],random_state=66)

    training_accuracy = []
    test_accuracy = []

    #try n_neighbors from 1 to 10
    neighbors_settings = range(1,11)

    for n_neighbors in neighbors_settings:
        #build the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        #record training set accuracy
        training_accuracy.append(knn.score(X_train, y_train))
        #record test set accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    plt.plot(neighbors_settings,training_accuracy, label="Training accuracy")
    plt.plot(neighbors_settings,test_accuracy, label="Test Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel(n_neighbors)
    plt.legend()
    plt.savefig('Knn_compare_model')
    plt.show()

    knn = KNeighborsClassifier(n_neighbors = 9)

    knn.fit(X_train, y_train)

    print('Accuracy of K-NN Classifier on training set : {:.2f}'.format(knn.score(X_train, y_train)))

    print('Accuracy of K-NN Classifier on test set : {:.2f}'.format(knn.score(X_test, y_test)))



def main():
    print("--------------Marvellous Infosystems by Piyush Khainar-------------- ")

    print("---Diabete Predictor Using K nearest neighbour-----------")

    DiabetePredictor()

if __name__ == "__main__":
    main()
