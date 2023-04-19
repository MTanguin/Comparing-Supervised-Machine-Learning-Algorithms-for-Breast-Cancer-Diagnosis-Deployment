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
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import seaborn as sns


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
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000,random_state=42)
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
        model_scores[name]['accuracy'] = accuracy_score(y_test, y_pred) # Add accuracy score to model_scores

    

    # Create a plot to compare all the scores
    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['accuracy','precision', 'recall', 'f1-score'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)



    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(
        go.Bar(x=df['Model'], y=df['accuracy'], name='Accuracy', marker_color='red'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['precision'], name='Precision', marker_color='blue'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['recall'], name='Recall', marker_color='green'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['f1-score'], name='F1-score', marker_color='purple'),
        row=2, col=2
    )

    fig.update_layout(height=800, title_text="Model Comparison")

    st.plotly_chart(fig)




#Run the Streamlit app
st.set_option('deprecation.showPyplotGlobalUse', False)
if __name__ == '__main__':
    app()