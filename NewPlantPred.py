import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib


# Load the dataset
crop_data = pd.read_csv("new_cropdata.csv")

# Split the data into features and target
X = crop_data.drop(columns=['Varieties of Crops grown'])
y = crop_data['Varieties of Crops grown']

# Encode the categorical feature
le = LabelEncoder()
X['Months'] = le.fit_transform(X['Months'])

# Impute missing values with the mean
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save the model to a file
joblib.dump(clf, "PlantWise_prediction_model.joblib")

# Load the saved model from a file
clf = joblib.load("PlantWise_prediction_model.joblib")



# Create a Streamlit app
st.title("Crop Prediction App")

# Create a text input for entering the month
month = st.text_input("Enter a month (e.g., January, February):")

# Create a button for predicting the crop
if st.button("Predict"):
    # Convert month to integer using the LabelEncoder
    month_int = le.transform([month])[0]

    # Predict the crops to be planted in the given month
    crop = clf.predict([[month_int]])

    # Display the predicted crop
    st.write(f"The crop to be planted in {month} is {crop[0]}")
