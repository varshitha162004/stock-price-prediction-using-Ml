import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import yfinance as yf
import io
from PIL import Image, ImageTk

# Function to train the model and generate predictions
def train_model(symbol, model_type='linear'):
    # Data Collection
    try:
        data = yf.download(symbol, start='2023-01-01', end='2024-11-14')
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
    except Exception as e:
        raise ValueError(f"Failed to download data for {symbol}: {e}")
    
    data = data[['Close']].dropna()
    
    # Feature Engineering
    data['Prediction'] = data['Close'].shift(-30)
    X = np.array(data.drop(['Prediction'], axis=1))[:-30]
    y = np.array(data['Prediction'])[:-30]
    
    # Model Selection
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'svr':
        model = SVR()
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X_train, y_train)
    
    # Model Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Predicting future stock prices
    future_predictions = model.predict(X[-30:])
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Actual Prices')
    plt.plot(data.index[-30:], future_predictions, label='Predicted Prices', linestyle='dashed')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save plot to a string buffer
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img = Image.open(img_buf)
    
    return img, mse, mae, r2

# Function to handle the prediction and update the GUI
def predict_stock():
    symbol = symbol_entry.get()
    model_type = model_choice.get()
    
    if not symbol:
        messagebox.showerror("Input Error", "Please enter a stock symbol")
        return
    
    try:
        img, mse, mae, r2 = train_model(symbol, model_type)
        img = ImageTk.PhotoImage(img)
        
        # Update the image label
        img_label.config(image=img)
        img_label.image = img
        
        # Update the evaluation metrics labels
        mse_label.config(text=f'Mean Squared Error: {mse:.4f}')
        mae_label.config(text=f'Mean Absolute Error: {mae:.4f}')
        r2_label.config(text=f'R-squared: {r2:.4f}')
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main application window
root = tk.Tk()
root.title("Stock Price Prediction")

# Create and place widgets
symbol_label = ttk.Label(root, text="Enter Stock Symbol:")
symbol_label.grid(row=0, column=0, padx=10, pady=10)

symbol_entry = ttk.Entry(root)
symbol_entry.grid(row=0, column=1, padx=10, pady=10)

# Model selection dropdown
model_choice_label = ttk.Label(root, text="Select Model:")
model_choice_label.grid(row=1, column=0, padx=10, pady=10)

model_choice = ttk.Combobox(root, values=['linear', 'random_forest', 'svr'])
model_choice.set('linear')  # Default model is Linear Regression
model_choice.grid(row=1, column=1, padx=10, pady=10)

predict_button = ttk.Button(root, text="Predict", command=predict_stock)
predict_button.grid(row=1, column=2, padx=10, pady=10)

img_label = ttk.Label(root)
img_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Labels to display evaluation metrics
mse_label = ttk.Label(root, text="Mean Squared Error: N/A")
mse_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

mae_label = ttk.Label(root, text="Mean Absolute Error: N/A")
mae_label.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

r2_label = ttk.Label(root, text="R-squared: N/A")
r2_label.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

# Run the application
root.mainloop()
