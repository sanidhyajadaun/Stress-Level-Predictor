# Importing necessary modules from the Flask framework
from flask import Flask, render_template, request, jsonify


import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Function to split data into training and testing sets
from sklearn.metrics import accuracy_score  # Function to calculate accuracy score
from sklearn.linear_model import LogisticRegression    # For Logistic Regression model

# Create an instance of the Flask class and name it 'app'
app = Flask(__name__)



@app.route('/')
def Stress_prediction_form():
    # The route decorator, '@app.route()', defines the URL path that this function will handle.
    # In this case, when a user accesses the root URL ('/'), this function will be executed.

    # The 'render_template' function is used to render an HTML template.
    # In this case, it will render the 'match_predict.html' template, which contains the form for stress prediction.
    return render_template('stress_predict.html')




@app.route('/predict', methods=['POST'])
 # The route decorator, '@app.route()', defines the URL path that this function will handle.
# In this case, when a POST request is made to the '/predict' URL, this function will be executed.
def predict():
     
    # Get data from user input
    snoring_rate = float(request.form['Snoring_rate'])
    respiration_rate = float(request.form['Respiration_rate'])
    body_temperature = float(request.form['Body_temperature'])
    limb_movement = float(request.form['Limb_movement'])
    blood_oxygen = float(request.form['Blood_oxygen'])
    eye_movement = float(request.form['Eye_movement'])
    sleeping_hours = float(request.form['Sleeping_hours'])
    heart_rate = float(request.form['Heart_rate'])

    # Reading the CSV file 'ipl.csv' and storing the data in a DataFrame called 'data'
    data = pd.read_csv(r'SaYoPillow.csv')
    
    # Identifying information about composition and potential data quality
    # data.info()

    # Renaming the columns of the DataFrame for better readability and understanding
    data.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', \
             'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']

    #checking for null values in the dataframe
    #data.isnull().sum()


    # Split the data into features (X) and the target variable (y)
    X = data.drop(['stress_level'], axis=1)
    y = data['stress_level']

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display the shapes of the training and testing sets
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

    # Create an instance of the Logistic_regression model
    model = LogisticRegression(max_iter=1000, C=0.1)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    #y_pred = model.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)


    # Creating a new DataFrame for the user input data
    new_data = pd.DataFrame([[snoring_rate, respiration_rate, body_temperature, limb_movement,
                          blood_oxygen, eye_movement, sleeping_hours, heart_rate]],
                        columns=X.columns)

    # Predict the stress level for the new data
    predicted_stress_level = model.predict(new_data)

    # Dictionary to map integer stress levels to human-readable labels
    stress_level_labels = {
        0: "Low/Normal",
        1: "Medium Low",
        2: "Medium",
        3: "Medium High",
        4: "High"
    }

    # Get the human-readable label for the predicted stress level
    predicted_stress_label = stress_level_labels[predicted_stress_level[0]]
    
    # Render the 'match_predict.html' template and pass the prediction result to display on the webpage
    return render_template('stress_predict.html', prediction=predicted_stress_label)




if __name__ == '__main__':
    # This block of code runs the Flask application when the script is executed directly.

    # The app.run() function starts the Flask development server.
    # - 'debug=True' enables the debug mode, which provides helpful error messages during development.
    # - 'host='0.0.0.0'' makes the app accessible from all network interfaces, allowing external access.
    app.run(debug=True, host='0.0.0.0')








