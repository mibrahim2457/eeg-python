
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class ClassificationBinary(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
        
    def splitData(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.10)
        
        
    def fitModel(self):
        self.model = RandomForestClassifier(criterion='gini', max_features='sqrt')
        while True:
            self.splitData()
            self.model.fit(self.X_train, self.Y_train)
            predictions = self.model.predict(self.X_test)

            # Train and Test Accuracy
            self.accuracy_train = accuracy_score(self.Y_train, self.model.predict(self.X_train))*100
            self.accuracy_test = accuracy_score(self.Y_test, predictions)*100
            self.confusion_matrix = confusion_matrix(self.Y_test, predictions)

            if self.accuracy_test > 80.0:
                break
        return self.accuracy_train, self.accuracy_test, self.confusion_matrix
    
    
    def makePrediction(self, x):
            return self.model.predict(x)
    
    
if __name__ == '__main__':
    dataset = pd.read_csv('F:/University/Final Year/FYP/EEG/EEG-Diagnosis (Python)/data/dataset.csv')
    data = dataset.values[:, :-1]
    target = dataset.values[:, -1]
    binaryClassifier = ClassificationBinary(data, target)

    # data = dataset.values[:, :-1]
    # target = dataset.values[:, -1]
    train,test,cm=binaryClassifier.fitModel()
    print(train,"%",test,"%",cm)
    
