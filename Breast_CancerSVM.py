#####################################################################
#Required Python Packages
#####################################################################

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

#####################################################################
#Funciton name : Marvellous
#Description : Load the data 
#Input : path of CSV file
#Output: Gives the data
#Author : Samidha Narendra Vaity
#Data : 04.112023
#####################################################################

def MarvellousSVM():
    #Load dataset
    cancer = datasets.load_breast_cancer()

    #print the names of the 13 features
    print("Features if the cancer dataset : ",cancer.feature_names)

    #print the label type of  cancer('maligant' bening)
    print("Labels of the cancer dataset : ",cancer.target_names)

    #print the cancer data features (top 5 records)
    print("First five records are : ")
    print(cancer.data[0:5])

    #split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)

    #Create a svcsvm classifier
    clf = svm.SVC(kernel = 'linear') #Linear kernel

    #Train the model using the training sets
    clf.fit(x_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(x_test)

    #Model Accuracy: how often is the classifier correct?
    print("Accuracy of the model is: ",metrics.accuracy_score(y_test,y_pred)*100)
def main():
    print("__________Marvellous Support Vector Machine__________")

    MarvellousSVM()
#####################################################################
#Application starter
#####################################################################
if __name__=="__main__":
    main()