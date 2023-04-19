# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define the Streamlit app
def app():
    # Set app title
    st.title("Breast Cancer Classification")

    # Read in the CSV file
    breast_cancer_data = pd.read_csv('Resources/breast_cancer_data.csv', usecols=lambda col: col != 'Unnamed: 32')

    # Drop the non-beneficial ID columns, 'id'
    breast_cancer_data = breast_cancer_data.drop(columns=['id'])

    # Display some data from the dataset
    st.write("Here's a peek at the dataset:")
    st.write(breast_cancer_data.head())

    # Get some basic statistics about the dataset
    st.write("Some basic statistics about the dataset:")
    st.write(breast_cancer_data.describe())

    # Split the dataset into training and testing sets
    X = breast_cancer_data.drop(columns=['diagnosis'])
    y = breast_cancer_data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define function 'evaluate_model()'
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

        st.write(f"{model_name} Classification Report:\n\n"
                 f"Recall: {round(recall, 2)}\n"
                 f"Precision: {round(prec, 2)}\n"
                 f"F1-score: {round(f1, 2)}\n\n")

    # Define models to be evaluated
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
    }

    # Evaluate each model and display results
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred, name)

    # Calculate the scores for each model
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        classification = classification_report(y_test, y_pred, output_dict=True)
        model_scores[name] = classification['macro avg']

    # Create a plot to compare all the scores
    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['precision', 'recall', 'f1-score'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)

    fig = px.bar(df, x='Model', y='precision', title='Model Comparison', barmode='group', height=500)
    fig.update_layout(xaxis_title='Model', yaxis_title='Precision')
    st.plotly_chart(fig)

#Run the Streamlit app
st.set_option('deprecation.showPyplotGlobalUse', False)
if __name__ == '__main__':
    app()