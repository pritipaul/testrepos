import streamlit as st
import numpy as np
import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("./Dataset/Dementia_Detection_clead_data.csv")

# Split into features and target
y = data['Group']
x = data.drop('Group', axis=1)

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Build the model
model = Sequential()
model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=8))
model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the trained model weights
model.load_weights("DementiaDetection_DL_Model.h5")

# Define class labels
class_labels = ['Non-Demented', 'Demented']

# Function to make predictions
def predict_dementia(features):
    # Preprocess the features
    processed_features = sc.transform([features])

    # Make predictions
    prediction = model.predict(processed_features)[0]

    # Get the predicted class label
    predicted_label = class_labels[int(np.round(prediction))]

    return predicted_label

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="game"
)

# Streamlit app code
def main():
    # Set the app title and description
    st.title("Dementia Classifier")
    st.write("Enter the patient's features and predict whether they are demented or non-demented.")

    # Feature inputs
    gender = st.radio("Gender", [0, 1])
    age = st.number_input("Age", min_value=0)
    educ = st.number_input("EDUC", min_value=0)
    ses = st.number_input("SES", min_value=0)
    mmse = st.number_input("MMSE", min_value=0)
    cdr = st.number_input("CDR", min_value=0.0, max_value=1.0, step=0.1)
    etiv = st.number_input("eTIV", min_value=0)
    nwbv = st.number_input("nWBV", min_value=0.0, max_value=1.0, step=0.001)

    # Make predictions if all features are provided
    if st.button("Predict"):
        features = [gender, age, educ, ses, mmse, cdr, etiv, nwbv]
        predicted_label = predict_dementia(features)
        st.write("Predicted Label:", predicted_label)

        # Store the prediction in the database
        cursor = db.cursor()
        insert_query = "INSERT INTO patient(gender, age, educ, ses, mmse, cdr, etiv, nwbv, predicted_label) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (gender, age, educ, ses, mmse, cdr, etiv, nwbv, predicted_label))
        db.commit()

# Run the app
if __name__ == '__main__':
    main()
