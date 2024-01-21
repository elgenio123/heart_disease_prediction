from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def create_model(X_train, y_train):

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model1 = RandomForestClassifier()
    model2 = SVC()
    model.fit(X_train, y_train)

    return model
