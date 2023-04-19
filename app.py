# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import seaborn as sns


# Define the Streamlit app
def app():
    # Set app title
    st.title("Supervised Machine Learning Algorithms for Breast Cancer Diagnosis")

    # Ask the user whether they want to use their own dataset for testing
    use_own_dataset = st.checkbox('Use your own dataset for testing?')

    # Read in the CSV file
    if use_own_dataset:
        uploaded_file = st.file_uploader('Choose a CSV file to upload', type='csv')
        if uploaded_file is not None:
            unseen_data = pd.read_csv(uploaded_file)
        else:
            st.warning('Please upload a CSV file.')
            return
    else:
        # use the default breast cancer dataset and test the models
        breast_cancer_data = pd.read_csv('Resources/breast_cancer_data.csv', usecols=lambda col: col != 'Unnamed: 32')
        breast_cancer_data = breast_cancer_data.drop(columns=['id'])
        unseen_data = breast_cancer_data.copy()

    # Display some data from the dataset
    st.write("Here's a peek at the dataset:")
    st.write(unseen_data.head())

    # Get some basic statistics about the dataset
    st.write("Some basic statistics about the dataset:")
    st.write(unseen_data.describe())

    # Allow users to choose test size and random state
    test_size = st.slider('Select test size', 0.1, 0.5, 0.2, 0.05)
    random_state = st.text_input('Enter random state', '42')

    # Split the dataset into training and testing sets
    X = unseen_data.drop(columns=['diagnosis'])
    y = unseen_data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    # Allow users to choose which models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Artificial Neural Network": MLPClassifier()
    }
    selected_models = st.multiselect('Select models to evaluate', list(models.keys()), default=list(models.keys()))

    # Define function 'evaluate_model()'
    def evaluate_model(test, pred, model_name):
        # Calculate accuracy score
        accuracy = accuracy_score(test, pred)
        st.write(f"{model_name} Accuracy: {round(accuracy*100,2)}%\n\n")

        # Generate a confusion matrix
        cm = confusion_matrix(test, pred)
        st.write(f"{model_name} Confusion Matrix:")
        st.write(cm)

        # Generate a classification report
        cr = classification_report(test, pred, labels=['B', 'M'])
        st.write(f"{model_name} Classification Report:")
        st.write(cr)

    # Evaluate the selected models
    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred, model_name)

      # Create a dictionary to store the evaluation metrics for each model
    model_metrics = {}

    # Evaluate the selected models and store their metrics in the dictionary
    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, labels=['B', 'M'], average='weighted')
        model_metrics[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1_score}

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(model_metrics, orient='index').reset_index()
    df = df.rename(columns={'index': 'Model'})

    # Melt the dataframe to long format
    df_melted = pd.melt(df, id_vars='Model', var_name='Metric', value_name='Score')

    # Create a bar chart using Plotly Express
    fig = px.bar(df_melted, x='Model', y='Score', color='Metric', barmode='group', title='Model Evaluation Metrics',
         range_y=[df_melted['Score'].min() - 0.1, max(df_melted['Score'].max(), 1)], 
         text='Score', 
         template='plotly_white'
         )

    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig.update_layout(xaxis={'categoryorder':'total descending'})

    # Adjust the spacing between the bars
    fig.update_layout(bargap=0.2)

    st.plotly_chart(fig)

#Run the Streamlit app
st.set_option('deprecation.showPyplotGlobalUse', False)
if __name__ == '__main__':
    app()