import streamlit as st
from streamlit_option_menu import option_menu
import csv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set up the page layout and title
st.set_page_config(layout="wide", page_title="AgriNexus")

# Custom CSS for a modern black theme
st.markdown("""
<style>
    .main {background-color: #1e1e1e; color: #ffffff; font-family: 'Arial', sans-serif;}
    .sidebar .sidebar-content {background-color: #000000; color: #ffffff;}
    .stButton>button {background-color: #28a745; color: white; font-size: 16px;}
    .stTextInput>div>input {color: #ffffff; background-color: #333333; border: 1px solid #ccc; padding: 8px;}
    .stTextInput>div>input::placeholder {color: #888888;}
    .stMarkdown {color: #ffffff;}
    .stDataFrame {background-color: #333333; border: 1px solid #ccc; color: #ffffff;}
    .stError {color: #ff4d4d;}
    .stWarning {color: #ffcc00;}
    .stSuccess {color: #28a745;}
    .title {color: #00ff00; text-align: center; padding: 10px 0;}
    .intro-box {background-color: #2c3e50; border-radius: 10px; padding: 20px; margin: 20px 0;}
</style>
""", unsafe_allow_html=True)

# Path to the CSV file
CSV_FILE = "user_data.csv"

# Sidebar menu
st.sidebar.title("Navigation")
with st.sidebar:
    menu_selection = option_menu("Main_Menu", ["Home", "Settings", "Dataset (CSV)", "Price Predictions"], default_index=0)

# Handle different menu selections
if menu_selection == "Home":
    st.markdown("<h1 class='title'>🌾 AgriNexus: Bridging Agriculture with Technology 🌾</h1>", unsafe_allow_html=True)

    # Display the introduction details in a styled box
    st.markdown("""
    <div class='intro-box'>
        <h2>Welcome to AgriNexus!</h2>
        <p>
            <strong>🌱 Challenge:</strong> Farmers face daunting challenges such as unpredictable crop diseases and fluctuating market prices, making it difficult to sustain their livelihoods.
        </p>
        <p>
            <strong>🌟 Our Solution:</strong> AgriNexus empowers farmers with cutting-edge tools that predict crop diseases and provide real-time market prices, fostering informed decision-making and boosting agricultural productivity.
        </p>
        <p>
            <strong>🎯 Our Goal:</strong> We aim to revolutionize the agricultural industry by delivering actionable insights, enabling farmers to thrive in a technology-driven world.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display the image
    st.image("img.png", use_column_width=True)

    # Create the login form with enhanced styling
    with st.form(key="login_form"):
        st.markdown("<h2 style='text-align: center;'>🔐 Login to Your Account</h2>", unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username")
        email = st.text_input("Email Address", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        submit_button = st.form_submit_button(label="Login")

    # Handle form submission
    if submit_button:
        if username and email and password:
            # Save to CSV
            file_exists = os.path.isfile(CSV_FILE)

            try:
                with open(CSV_FILE, 'a', newline='') as csvfile:
                    fieldnames = ['Username', 'Email Address', 'Password']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # Write header if file is empty
                    if not file_exists:
                        writer.writeheader()

                    writer.writerow({'Username': username, 'Email Address': email, 'Password': password})

                # Confirm successful saving
                st.success(f"🎉 Welcome, {username}! Your data has been saved successfully.")

            except Exception as e:
                st.error(f"⚠️ An error occurred while saving data: {e}")
        else:
            st.warning("⚠️ Please fill out all fields.")

elif menu_selection == "Settings":
    st.title("⚙️ Settings")
    st.write("Customize your experience here...")

elif menu_selection == "Dataset (CSV)":
    st.title("📊 User Data")
    if os.path.isfile(CSV_FILE):
        st.write("Here is the data stored in the CSV file:")

        # Read CSV with the correct headers
        try:
            df = pd.read_csv(CSV_FILE)

            # Check for duplicate columns and remove them
            df = df.loc[:, ~df.columns.duplicated()]

            st.dataframe(df)
        except pd.errors.EmptyDataError:
            st.warning("⚠️ The CSV file is empty.")
            df = pd.DataFrame(columns=['Username', 'Email Address', 'Password'])
            st.dataframe(df)
        except pd.errors.ParserError:
            # Handle the case where there are no headers in the CSV file
            st.warning("⚠️ CSV file seems to have no headers. Assuming default headers.")
            df = pd.read_csv(CSV_FILE, header=None, names=['Username', 'Email Address', 'Password'])
            st.dataframe(df)
        except Exception as e:
            st.error(f"⚠️ An error occurred while reading the CSV file: {e}")
    else:
        st.warning("⚠️ No data found. Please login to add some data.")

elif menu_selection == "Price Predictions":
    st.title("📈 Commodity Price Predictions")

    # Load the CSV file with historical data
    try:
        df = pd.read_csv("predict_data.csv")

        # Check for duplicate columns and remove them
        df = df.loc[:, ~df.columns.duplicated()]

        # Extract years from the columns and set 'Commodity' as index
        years = df.columns[1:]  # Extract years from columns
        df = df.set_index('Commodity')

        # Display the commodity selection dropdown
        commodity = st.selectbox("Select Commodity", df.index)

        # Prepare for scaling and model training
        future_years = ['2025-26', '2026-27']  # Future years to predict

        if commodity:
            # Prepare data
            X = np.arange(len(df.columns)).reshape(-1, 1)  # years as features
            y = df.loc[commodity].values  # prices as target

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X_scaled, y)

            # Predict for 2025-26 and 2026-27
            years_to_predict = np.array([[len(df.columns)], [len(df.columns) + 1]])  # 2025-26 and 2026-27
            years_to_predict_scaled = scaler.transform(years_to_predict)
            predicted_prices = model.predict(years_to_predict_scaled)

            # Prepare predictions DataFrame
            predictions = pd.DataFrame(
                {
                    'Year': future_years,
                    'Predicted Price': predicted_prices
                }
            )

            # Display the DataFrame
            st.write(f"Predictions for {commodity}:")
            st.dataframe(predictions)

            # Option to download the predictions as a CSV file
            st.download_button(
                label="Download Predictions",
                data=predictions.to_csv(index=False),
                file_name=f"{commodity}_predictions_2025_2026.csv"
            )
        else:
            st.warning("⚠️ Please select a commodity.")
    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the predictions: {e}")
