import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
#from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from Models import *
from autoformer import autoformer_model

if __name__ == "__main__":
    data = pd.read_csv("/content/AAPL_2006-01-01_to_2018-01-01 (1).csv", parse_dates=['Date'])
    data.set_index('Date', inplace=True)

    # We will use 'Close' prices for forecasting
    close_prices = data['Close']

    # Split the data into training and testing sets
    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices[:train_size], close_prices[train_size:]


    #Baseline Modelling
    # Make predictions
    predictions = model_fitted.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

    # Evaluate the model using RMSE
    rmse = sqrt(mean_squared_error(test, predictions))

    print(f"RMSE: {rmse}")

    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    #RNN Prediction
    RNNpredictions = model.predict(X_test)
    RNNpredictions = scaler.inverse_transform(RNNpredictions)

    #LSTM Prediciton
    predictions = lst_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    #Autoformer Model
    # Evaluate the model
    autoformer_model.eval()
    with torch.no_grad():
        # Convert X_test to a PyTorch tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        predictions = autoformer_model(X_test_tensor)

        # Convert y_test to a PyTorch tensor if it's not already
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device) if not isinstance(y_test, torch.Tensor) else y_test

        test_loss = criterion(predictions.squeeze(), y_test_tensor.squeeze())
        print(f"Test Loss: {test_loss.item():.4f}")

    # Convert predictions and actual values back to original scale
    predictions = predictions.cpu().numpy() # Move predictions to CPU before converting to NumPy
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))