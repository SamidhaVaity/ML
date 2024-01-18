import numpy as numpy
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):

    # Step1: Load data
    data = pd.read_csv(data_path, index_col = 0)

    print("Size of Actual dataset", len(data))

    #step 2 : clean, prepare & manipulate data
    feature_names = ['Whether','Temperature']

    print("Names of Features", feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    #creating labelEncoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers
    weather_encoded = le.fit_transform(whether)
    print (weather_encoded)

    # Converting string labels into numbers.
    temp_encoded = le.fit_transform(Temperature)
    label =le.fit_transform(play)

    print(temp_encoded)

    #combining weather and temp into single listof tuples
    features = list(zip(weather_encoded, temp_encoded))

    #Step 3 : Train Data
    model = KNeighborsClassifier(n_neighbors =3)

    #Train the model using the training sets
    model.fit(features,label)

    # Step4 : Test data
    predicted = model.predict([[1,0]])
    print(predicted)




def main():
    print("---Marvellous Infosystems by Piyush Khairnar----")

    print("Machine Leaning Application")

    print("Play predictor application using k Nearest knighbor algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__ =="__main__":
    main()