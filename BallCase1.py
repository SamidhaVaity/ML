from sklearn import *

def main():
    print("Ball Classification case study")

    # /Load the Data
    BallFeatures = [[35,"Rough"],[47,"Rought"],[90,"Smooth"],[48."Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],[96,"smooth"],[43,"Rough"],[110,"Smooth"],[35,"Rough"],[95,"Smooth"],[35,"Rough"]]
   
    Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket","Tennis","Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket","Tennis"]                 

    obj = tree.DecisionTreeClassifier()  # Decide algorithm

    obj = obj.fit(BallFeatures,Labels)  # Train the model

    print(obj.predict([[36,"Rought"],[96,"Smooth"]]))   # Test the model            
                    
                    
if __name__ == "__main__":
    main()