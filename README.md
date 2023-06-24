#### Project_4_Deployment_Test

Allows users to test and evaluate different machine learning algorithms on a breast cancer dataset to diagnose whether a patient has malignant or benign cancer. The evaluation metrics are displayed to the user in an interactive bar chart.

## HOW IT WORKS

- This is a web application that allows users to test different machine learning algorithms on a breast cancer dataset to diagnose whether a patient has malignant or benign cancer.

- The application uses Streamlit, a Python library that helps to build interactive web applications. 

- The application starts by asking the user if they want to use their own dataset for testing, and if so, they are prompted to upload a CSV file. If the user chooses not to use their own dataset, a default breast cancer dataset is used for testing the models.

- The dataset is split into training and testing sets, and the user can choose which machine learning algorithms they want to test. The selected algorithms are then evaluated and the results are displayed to the user.

- The evaluation metrics for each model are stored in a dictionary and then converted into a Pandas DataFrame. The DataFrame is then melted into long format, and a bar chart is created using Plotly Express to display the evaluation metrics for each model.


