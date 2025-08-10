# stock-price-prediction-using-Ml

The application allows the user to:

Enter a stock symbol (e.g., AAPL, MSFT, TSLA).

Fetch historical data (from Yahoo Finance via yfinance).

Train a Linear Regression model to predict stock prices 30 days into the future.

Display a graph comparing the actual and predicted prices.

Show the model's Mean Squared Error (MSE) as a performance metric.

This Python script is a graphical application that allows users to predict future stock prices using a simple machine learning model. Built with the Tkinter library, the application provides a user-friendly interface where a user can input a stock symbol (such as AAPL for Apple or MSFT for Microsoft) and click a button to initiate the prediction process. Upon clicking "Predict", the application retrieves historical stock data from Yahoo Finance using the yfinance library, specifically focusing on the stock’s closing prices from January 1, 2023, to November 14, 2024. It then prepares the data by shifting the closing prices 30 days forward, effectively turning the task into a supervised learning problem. The script splits the data into training and testing sets and fits a linear regression model from the scikit-learn library to this data. Once the model is trained, it predicts the closing prices for the next 30 days. These predictions are plotted alongside the actual closing prices using matplotlib, and the resulting chart is displayed within the application using Pillow to render the image in the Tkinter GUI. Additionally, the model’s performance is measured using Mean Squared Error (MSE), which is also shown to the user. This tool serves as a simple yet interactive demonstration of applying machine learning to financial data, although its use of a basic linear regression model may not capture the complexities of stock market behavior.




