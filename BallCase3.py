from sklearn import tree

def MarvellousML(weight, surface):

    BallFeatures = [[35,1], [47,1], [90,0], [48,1], [90,0], [35,1], [92,0], [35,1], [35,1], [35,1], [96,0], [43,1], [110,0], [35,1], [95,0]]
   
    Names = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    clf = tree.DecisionTreeClassifier()  # Decide algorithm

    clf = clf.fit(BallFeatures,Names)  # Train the model

    result = clf.predict([[weight,surface]])
    
    if result == 1:
        print("Your object looks like Tennis Ball")
    elif result == 2:
        print("Your object looks like Cricket Ball")



def main():
    print("-----------------Marvellous Infosystem-----------------")

    print("Enter weight of object")
    weight = input()
    
    print("What is the surface type of your object rough or smooth")
    surface = input()

    if surface.lower()== "rough":
        surfacce = 1
    elif surface.lower()== "smooth":
        surfacce = 0
    else:
        print("Error :Wrong Input")
        exit()

    
    MarvellousML(weight,surface)
                    
if __name__ == "__main__":
    main()