# start by installing the Streamlit library using pip

# import dependencies
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#define function 'evaluate_model()'

def evaluate_model(test, pred, model_name):
    # Calculate accuracy score
    accuracy = accuracy_score(test, pred)
    st.write(f"{model_name} Accuracy: {round(accuracy*100,2)}%\n\n")

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(test, pred)
    st.write(f"{model_name} Confusion Matrix:\n\n{conf_matrix}\n\n")

    # Generate a classification report
    classification = classification_report(test, pred, output_dict=True)
    recall = classification['macro avg']['recall']
    f1 = classification['macro avg']['f1-score']
    prec = classification['macro avg']['precision']

    st.write(f"{model_name} Classification Report:\n\n{classification_report(test, pred)}\n\n")

    # Plot the confusion matrix as a heatmap
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"{model_name} Confusion Matrix")
    st.pyplot()

    return prec, accuracy, recall, f1

#load in your data and perform any necessary data cleaning and preprocessing

# Read in the CSV file
breast_cancer_data = pd.read_csv('Resources/breast_cancer_data.csv', usecols=lambda col: col != 'Unnamed: 32')

# Drop the non-beneficial ID columns, 'id'
breast_cancer_data = breast_cancer_data.drop(columns = ['id'])

# Define the Streamlit app
def app():
    # Set app title
    st.title("Breast Cancer Classification")

    # Display some data from the dataset
    st.write("Here's a peek at the dataset:")
    st.write(breast_cancer_data.head())

    # Get some basic statistics about the dataset
    st.write("Some basic statistics about the dataset:")
    st.write(breast_cancer_data.describe())

    # Visualize some data from the dataset
    st.write("Here's a visualization of the 'radius_mean' feature:")
    sns.histplot(data=breast_cancer_data, x='radius_mean', bins=30)
    st.pyplot()

    # Split the dataset into training and testing sets
    X = breast_cancer_data.drop(columns=['diagnosis'])
    y = breast_cancer_data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate some machine learning models
    st.write("Training and evaluating some machine learning models...")
    
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    evaluate_model(y_test, lr_pred, "Logistic Regression")

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    evaluate_model(y_test, dt_pred, "Decision Tree")

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    evaluate_model(y_test, rf_pred, "Random Forest")

    # Support Vector Machine
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    evaluate_model(y_test, svm_pred, "Support Vector Machine")

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    evaluate_model(y_test, knn_pred, "K-Nearest Neighbors")

    # Artificial Neural Network
    ann_model = Sequential()
    ann_model.add(Dense(16, input_dim=30, activation='relu'))
    ann_model.add(Dense(8, activation='relu'))
    ann_model.add(Dense(1, activation='sigmoid'))
    ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ann_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    ann_pred = ann_model.predict(X_test)
    ann_pred = np.where(ann_pred > 0.5, 1, 0)
    evaluate_model(y_test, ann_pred, "Artificial Neural Network")