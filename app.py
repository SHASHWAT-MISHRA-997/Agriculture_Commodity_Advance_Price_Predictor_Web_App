import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import base64
from statsmodels.tsa.seasonal import seasonal_decompose
from fpdf import FPDF
import streamlit as st
import logging
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import tempfile
import os


current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, "Agriculture.jpeg")
try:
    st.sidebar.image(logo_path, width=400)
    
except Exception as e:
    
    
    print(f"Logo file exists: {os.path.exists(logo_path)}")

# Set Streamlit app title and description
st.title("🌾 AI-ML Based Commodity Price Prediction System for Agri-Horticultural Markets:")

# Custom CSS for background hover effect and button colors
st.markdown(""" 
    <style>
        body {
            background: linear-gradient(135deg, rgba(255,0,0,0.7), rgba(0,0,255,0.7));
            transition: background-color 0.5s ease;
        }
        .stButton > button:hover {
            background-color: rgba(255, 165, 0, 0.8);
            color: white;
        }
        .stButton > button {
            background-color: rgba(0, 255, 0, 0.7);
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 16px;
            transition: background-color 0.3s, transform 0.3s;
        }
        .stButton > button:active {
            transform: scale(0.95);
        }
        .creator-link {
            color: black;
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            display: block;  
            text-align: center;  
            margin-top: 10px;   
        }
        .creator-link:hover {
            background-color: white;  
            color: red;              
            padding: 5px;           
            border-radius: 5px;     
        }
    </style>
""", unsafe_allow_html=True)

# Creator link at the top of the sidebar
st.markdown(
    '<a href="https://www.linkedin.com/in/sm980/" class="creator-link">Created by SHASHWAT MISHRA</a>',
    unsafe_allow_html=True
)


st.markdown("""This app provides intelligent predictions and insights into agricultural commodity prices  designed for businesses and government bodies.
Use this tool to analyze historical trends, predict prices and generate comprehensive reports.
""")

# Sidebar Instructions
st.sidebar.header("📋 Instructions : ")
st.sidebar.markdown("""
1. Select a **State, District, Market ** and **Commodity**.
2. Choose a **Date** for prediction.
3. Click **Predict Price** for results.
4. Use tools like **Price Trend**, **Correlation Heatmap** , **Price Comparison** and **Volatility Analysis**.
5. Export data to **CSV** .
""")

# Load the CSV file
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"{file_path} loaded successfully.")
        data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], format="%d-%m-%Y")  # Ensure 'Arrival_Date' is parsed as datetime
        return data
    except FileNotFoundError:
        logging.error(f"{file_path} not found. Please check the file path.")
        return pd.DataFrame()  # Empty DataFrame to prevent further errors

commodity_data = load_data("Price_Agriculture_commodities_Week.csv")

if commodity_data.empty:
    st.error("Error: No data available. Please check the dataset.")
else:
    st.success("Data loaded successfully.")

# Dropdown lists for user selection
states = commodity_data['State'].unique()

# Streamlit user selections
st.subheader("🔍 Predict Commodity Price : ")
st.write("Select the commodity details to get actual or predicted prices.")

selected_state = st.selectbox("Select a State", states)
filtered_districts = commodity_data[commodity_data['State'] == selected_state]['District'].unique()
selected_district = st.selectbox("Select a District", filtered_districts)

filtered_markets = commodity_data[
    (commodity_data['State'] == selected_state) &
    (commodity_data['District'] == selected_district)
]['Market'].unique()
selected_market = st.selectbox("Select a Market", filtered_markets)

filtered_commodities = commodity_data[
    (commodity_data['State'] == selected_state) &
    (commodity_data['District'] == selected_district) &
    (commodity_data['Market'] == selected_market)
]['Commodity'].unique()
selected_commodity = st.selectbox("Select a Commodity", filtered_commodities)

selected_date = st.date_input("Select Date")

# Filter data based on user selection
filtered_data = commodity_data[
    (commodity_data['State'] == selected_state) &
    (commodity_data['District'] == selected_district) &
    (commodity_data['Market'] == selected_market) &
    (commodity_data['Commodity'] == selected_commodity)]

# Train models for each commodity based on historical data
def train_model():
    if filtered_data.empty:
        logging.error("No data available for training models.")
        return None, None, None
    
    # Prepare the dataset for the selected commodity
    df = filtered_data[['Arrival_Date', 'Modal Price']].dropna()
    df['Arrival_Date'] = df['Arrival_Date'].map(pd.Timestamp.toordinal)

    if len(df) < 2:
        logging.error("Insufficient data for model training.")
        st.error("Insufficient data for model training. Please provide more data.")
        return None, None, None
    
    # Split data into features and target
    X = df[['Arrival_Date']]
    y = df['Modal Price']

    # Use K-Fold cross-validation for small datasets
    kf = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
    model = LinearRegression()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model()

# Function to predict price if exact date is not available
def predict_price(date):
    if model:
        try:
            date_ordinal = pd.to_datetime(date).toordinal()
            prediction = model.predict(np.array([[date_ordinal]]))
            return prediction[0]
        except Exception as e:
            logging.error(f"Error predicting price: {e}")
            st.error(f"Error predicting price: {e}")
            return None
    return None

# Function to calculate prediction confidence interval
def get_prediction_confidence():
    if model:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        confidence_interval = 1.96 * math.sqrt(mse)  # 95% confidence interval
        return confidence_interval
    return None

predicted_price = None  # Initialize variable for storing predicted price

if st.button("Predict Price"):
    try:
        selected_date = pd.to_datetime(selected_date)

        # Filter the dataset based on the selected date
        date_filtered_data = filtered_data[filtered_data['Arrival_Date'] == selected_date]

        if not date_filtered_data.empty:
            modal_price = date_filtered_data['Modal Price'].values[0]
            st.success(f"Actual Modal Price on {selected_date.strftime('%Y-%m-%d')}: ₹{modal_price}")
            predicted_price = None  # Clear prediction
        else:
            predicted_price = predict_price(selected_date)
            if predicted_price is not None:
                confidence_interval = get_prediction_confidence()
                st.success(f"Predicted Modal Price on {selected_date.strftime('%Y-%m-%d')}: ₹{predicted_price:.2f} ± ₹{confidence_interval:.2f}")
            else:
                st.error("Prediction model is not available.")
    except Exception as e:
        st.error(f"Error: {e}")

# Add Date Range Selection
st.subheader("📅 Predict Prices for Date Range : ")
start_date = st.date_input("Select Start Date", value=datetime.date.today())
end_date = st.date_input("Select End Date", value=datetime.date.today() + datetime.timedelta(days=30))

# Validate date range
if start_date > end_date:
    st.error("End Date must be after Start Date.")
else:
    # Function to predict prices for a range of dates
    def predict_prices_for_date_range(start_date, end_date):
        dates = pd.date_range(start_date, end_date)
        predictions = []
        for date in dates:
            prediction = predict_price(date)
            if prediction is not None:
                predictions.append({"Date": date, "Predicted Price": prediction})
        return pd.DataFrame(predictions)

    # Display price predictions for the selected date range
    if st.button("Predict Prices for Date Range"):
        try:
            price_predictions = predict_prices_for_date_range(start_date, end_date)
            if not price_predictions.empty:
                st.write(f"Predicted Prices from {start_date} to {end_date}:")
                st.dataframe(price_predictions)

                # Plot predictions
                plt.figure(figsize=(12, 6))
                plt.plot(price_predictions['Date'], price_predictions['Predicted Price'], marker='o', linestyle='-')
                plt.title('Price Predictions Over Time')
                plt.xlabel('Date')
                plt.ylabel('Predicted Price (₹)')
                plt.xticks(rotation=45)
                plt.grid(True)
                img = BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                st.image(img, caption='Price Predictions Plot')
            else:
                st.error("No predictions available for the selected date range.")
        except Exception as e:
            st.error(f"Error predicting prices: {e}")
# Visualizations and trend analysis
st.subheader("📈 Price Trend and Analysis : ")
st.write("Visualize Price Trends, Correlation Heatmap, Price Comparison and Volatility for better decision-making.")

def plot_prices():
    df = filtered_data[['Arrival_Date', 'Modal Price']].dropna()
    plt.figure(figsize=(10,6))
    plt.plot(df['Arrival_Date'], df['Modal Price'], label=selected_commodity)
    plt.title(f"{selected_commodity} Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Modal Price (₹)")
    plt.xticks(rotation=45)
    plt.legend()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

def show_correlation_heatmap():
    df = commodity_data.pivot_table(index='Arrival_Date', columns='Commodity', values='Modal Price')
    
    # Compute the correlation matrix
    corr = df.corr()
    
    # Increase the figure size for better readability
    plt.figure(figsize=(14, 10))  # Larger figure size
    
    # Use a color map with high contrast for clarity
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(
        corr,
        annot=True,  # Display correlation coefficients
        cmap=cmap,
        linewidths=0.5,
        fmt=".2f",  # Format the annotations to 2 decimal places
        vmin=-1,  # Set min value for color scale
        vmax=1,   # Set max value for color scale
        center=0,  # Center the color scale at 0
        cbar_kws={'shrink': 0.8}  # Adjust color bar size
    )
    
    # Improve readability of x and y labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Correlation Heatmap of Commodity Prices", size=16)
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the plot to free memory
    return img


def price_comparison():
    df = filtered_data[['Arrival_Date', 'Modal Price']].dropna()
    df.set_index('Arrival_Date', inplace=True)
    df = df.resample('W').mean()  # Resample data to weekly frequency
    
    historical_avg = df['Modal Price'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Modal Price'], label='Weekly Average Price')
    plt.axhline(y=historical_avg, color='r', linestyle='--', label='Historical Average Price')
    plt.title(f'{selected_commodity} Weekly Average Prices')
    plt.xlabel('Date')
    plt.ylabel('Average Price (₹)')
    plt.legend()
    plt.xticks(rotation=45)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

st.subheader("📉 Price Trend : ")
if st.button("Show Price Trend"):
    st.image(plot_prices(), caption="Price Trend")

st.subheader("🔍 Correlation Heatmap : ")
if st.button("Show Correlation Heatmap"):
    st.image(show_correlation_heatmap(), caption="Correlation Heatmap")

st.subheader("📊 Price Comparison : ")
if st.button("Show Price Comparison"):
    st.image(price_comparison(), caption="Price Comparison")

# Volatility Analysis
st.subheader("📊 Volatility Analysis : ")
st.write("Analyze the price volatility of the selected commodity.")

def calculate_volatility():
    df = filtered_data[['Arrival_Date', 'Modal Price']].dropna()
    df.set_index('Arrival_Date', inplace=True)
    df = df.resample('W').mean()  # Resample to weekly
    volatility = df['Modal Price'].std()
    return volatility

if st.button("Calculate Volatility"):
    volatility = calculate_volatility()
    st.write(f"Price Volatility: {volatility:.2f}")

# Generate CSV Report
st.subheader("📄 Export Data : ")
if st.button("Export Data to CSV"):
    df = filtered_data[['Arrival_Date', 'Modal Price']]
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='commodity_prices.csv',
        mime='text/csv'
    )
