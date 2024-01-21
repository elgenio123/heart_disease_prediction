import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
from model import create_model
from dataset import replace_missing_with_mode 

def main():

    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets
    data = pd.concat([X, y], axis=1)
  

    data = replace_missing_with_mode(data)
    print(data)

    X = data.drop('num', axis=1)
    y = data['num']
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(X_train, y_train)
    predictions = model.predict(X_test)
    #print(predictions)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

if __name__== "__main__":
    main()