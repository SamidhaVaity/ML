from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def MarvellousCalculateAccuracyDecisionTree():
    iris = load_iris()

    data = iris.data
    target = iris.target

    data_train, data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

    classifier = tree.DecisionTreeClassifier()
    
    classifier.fit(data_train, target_train)

    predications = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,predictions)

    return Accuracy

def MarvellousCAlculateAccuracyKNeighbor():
    iris = load_iris()

    data = iris.data
    target = iris.target

    data_train, data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

    classifier = KneighborsClassifier()

    classifier.fit(data_train, target_train)

    predications = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,predictions)

    return Accuracy



def main():
    Accuracy = MarvellousCalculateAccuracyDecisionTree()
    Print("Accuracy of classification algorithm with Decision Tree classifier is ",Accuracy*100,"% ")

    Accuracy = MarvellousCAlculateAccuracyKNeighbor()
    Print("Accuracy of classification algorithm with K Neighbor classifier is ",Accuracy*100,"% ")

if __name__=="__main__":
    main()