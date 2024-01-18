#####################################################################
#Required Python Packages
#####################################################################
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#####################################################################
#File Paths
#####################################################################
INPUT_PATH = "breast-cancer-wiscosin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"

#####################################################################
#Header
#####################################################################
HEADERS = ["CodeNumber","ClumpThickness","UniformityCellSize","MarginalAdhesion","SingleEpithelialCellSize","BareNuclie","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]

#####################################################################
#Funciton name : read_data
#Description : Read the data into pandas dataframe
#Input : path of CSV file
#Output: Gives the data
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def read_data(path):
    data = pd.read_csv(path)
    return data

#####################################################################
#Funciton name : add_headers
#Description : Add the headers to the dataset
#Input : dataset
#Output: updated dataset
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

#####################################################################
#Funciton name : data_file_to_csv
#Input : Nothing
#Output: Write the data to CSV
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def data_file_to_csv():
    #Headers
    headers = ["CodeNumber","ClumpThickness","UniformityCellSize","MarginalAdhesion","SingleEpithelialCellSize","BareNuclie","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]
    #Load the dataset into pandas data freame
    dataset = read_data(INPUT_PATH)
    #Add the headers to the loaded data set
    dataset = add_headers(dataser, headers)
    #Save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH, index =False)
    print("File saved ...!")

#####################################################################
#Funciton name : split_dataset
#Description : Split the dataset with train_percentage
#Input : Nothing
#Output: Write the data to CSV
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def split_dataset(dataset,train_percentage, feature_headers, target_header):
    #Split dataset into train & test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers],dataset[target_header], train_size = train_percentage)
    return train_x,test_x,train_y,test_y

#####################################################################
#Funciton name : handel_missing_values
#Description : Filter missing values from the dataset
#Input : Dataset with missing values
#Output: Dataset by remocking missing values
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def handel_missing_values(dataset, missing_values_header,missing_label):
    return dataset[dataset[missing_values_header]!=missing_label]

#####################################################################
#Funciton name : random_forest_classifier
#Description : To train the random forest classifier with features and target data
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

#####################################################################
#Funciton name : dataset_statictics
#Description : Basic statistivs of the dataset
#Input : Dataset
#Output : Description of dataset
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def dataset_statistics(dataset):
    print(dataset.describe())

#####################################################################
#Funciton name : main
#Description : main function from where execution starts
#Author : Samidha Narendra Vaity
#Data : 31/10/2023
#####################################################################

def main():
    #Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    #Get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    #Filter missing values
    dataset = handel_missing_values(dataset, HEADERS[6],'?')
    train_x,test_x,train_y,test_y = split_dataset(dataset, 0.7,HEADERS[1:-1],HEADERS[-1])

    #Train and test dataset size details
    print("Train_x Shape ::",train_x.shape)
    print("Train_y Shape ::",train_y.shape)
    print("Test_x Shape ::",test_x.shape)
    print("Test_y Shape ::",test_x.shape)

    #Create random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model ::",trained_model)
    prediction = trained_model.predict(test_x)

    for i in range(0, 205):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i],prediction[i]))

    print("Train Accuracy ::", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy ::", accuracy_score(test_y, predictions))
    print("Confusion Matric ", confusion_matrix(test_y, predictions))

#####################################################################
#Application starter
#####################################################################
if __name__=="__main__":
    main()