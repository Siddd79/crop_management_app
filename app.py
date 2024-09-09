import streamlit as st
import sqlite3
import pandas as pd
# import bcrypt
import os
import json
from PIL import Image
import numpy as np
# import tensorflow as tf

# Initialize SQLite database connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    return conn

# Create tables if not already present
def initialize_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        full_name TEXT,
                        email TEXT UNIQUE,
                        phone TEXT UNIQUE,
                        username TEXT UNIQUE,
                        password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS crop_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        crop_name TEXT,
                        health_status TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        alert_message TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id))''')
        conn.commit()

initialize_db()

# Load the pre-trained model and class indices
# working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# model = tf.keras.models.load_model(model_path)
# class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Functions for image classification
# def load_and_preprocess_image(image, target_size=(224, 224)):
#     img = Image.open(image)
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array.astype('float32') / 255.
#     return img_array

# def predict_image_class(model, image, class_indices):
#     preprocessed_img = load_and_preprocess_image(image)
#     predictions = model.predict(preprocessed_img)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class_name = class_indices[str(predicted_class_index)]
#     return predicted_class_name

# User registration
def register():
    st.title("Register")
    full_name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Register"):
        if password == confirm_password:
            # hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            hashed_password = password  # Using plain password for now since bcrypt is commented out
            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO users (full_name, email, phone, username, password) VALUES (?, ?, ?, ?, ?)",
                              (full_name, email, phone, username, hashed_password))
                    conn.commit()
                    st.success("User registered successfully")
            except sqlite3.IntegrityError as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("Passwords do not match")

# User login
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            # if result and bcrypt.checkpw(password.encode(), result[0]):
            if result and password == result[0]:  # Using plain password comparison since bcrypt is commented out
                st.success("Logged in successfully")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
            else:
                st.error("Invalid credentials")

# Dashboard
def dashboard():
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        username = st.session_state["username"]
        st.title("Dashboard")
        st.write(f"Welcome, {username}")

        st.subheader("Manage Crop Health")
        crop_name = st.text_input("Crop Name")
        health_status = st.text_input("Health Status")
        if st.button("Add Crop Health"):
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("INSERT INTO crop_health (user_id, crop_name, health_status) VALUES ((SELECT id FROM users WHERE username = ?), ?, ?)",
                          (username, crop_name, health_status))
                conn.commit()
                st.success("Crop health added successfully")

        st.subheader("Manage Alerts")
        alert_message = st.text_input("Alert Message")
        if st.button("Add Alert"):
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("INSERT INTO alerts (user_id, alert_message) VALUES ((SELECT id FROM users WHERE username = ?), ?)",
                          (username, alert_message))
                conn.commit()
                st.success("Alert added successfully")

        st.subheader("Your Crop Health Records")
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT crop_name, health_status FROM crop_health WHERE user_id = (SELECT id FROM users WHERE username = ?)", (username,))
            rows = c.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["Crop Name", "Health Status"])
                st.table(df)
            else:
                st.write("No crop health records found")

        st.subheader("Your Alerts")
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT alert_message FROM alerts WHERE user_id = (SELECT id FROM users WHERE username = ?)", (username,))
            alerts = c.fetchall()
            if alerts:
                for alert in alerts:
                    st.write(alert[0])
            else:
                st.write("No alerts found")
    else:
        st.error("Please log in to view the dashboard.")

# Plant Disease Classifier
def plant_disease_classifier():
    st.title("Plant Disease Classifier")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        # with col2:
        #     if st.button('Classify'):
        #         prediction = predict_image_class(model, uploaded_image, class_indices)
        #         st.success(f'Prediction: {str(prediction)}')

# Main navigation
def main():
    st.sidebar.title("Navigation")
    menu = ["Login", "Register", "Dashboard", "Plant Disease Classifier"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Add the logout button in the sidebar
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.success("You have been logged out.")

    if choice == "Register":
        register()
    elif choice == "Login":
        login()
    elif choice == "Dashboard":
        dashboard()
    elif choice == "Plant Disease Classifier":
        plant_disease_classifier()

# Run the app
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None

    main()
