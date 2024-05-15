# Name-Anirudha Kapileshwari
# Spring 2024 Introduction to Machine Learning (CS-484-01)
# prof Shouvik Roy


# Assignment 1 CS 484
# Decision Tree Learning

# Inspect the dataset titled lab01_dataset_2.csv which also has a mixture of numerical and categorical data. 
# For this problem, you will use decision tree classifiers for supervised learning. In particular, you will be 
# using the functionalities of the sklearn.tree library. The classification task using sklearn libraries work only on 
# numerical-valued attributes, and not on categorical ones. (What to do now? Hint: Look up One-hot Encoding and Integer Encoding). 
# Here are all the required tasks –


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def upload_data(file_path):
    return pd.read_csv(file_path)

def one_hot_encode(dataset, columns_to_encode):
    return pd.get_dummies(dataset, columns=columns_to_encode)

def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def check_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    return results, accuracy

def construct_tree(clf, feature_names, class_names):
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()

def main():
    dataset = upload_data('lab01_dataset_2.csv')

    columns_to_enc = ['Sex', 'BP', 'Na_to_K', 'Cholesterol']
    dataset_enc = one_hot_encode(dataset, columns_to_enc)

    X = dataset_enc.drop('Output', axis=1)
    y = dataset_enc['Output']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    clf = train_decision_tree(X_train, y_train)

    results, accuracy = check_model(clf, X_test, y_test)

    print(results)

    print(f'Accuracy: {accuracy}')

    feature_names = list(X.columns)
    class_names = list(dataset['Output'].unique())
    construct_tree(clf, feature_names, class_names)

if __name__ == "__main__":
    main()
