from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Dense



# Fit the AutoReg model
model = AutoReg(train, lags=730)
model_fitted = model.fit()

#RNN Model
model = Sequential([
    SimpleRNN(200, return_sequences=True, input_shape=(seq_length, 1)),
    SimpleRNN(100, return_sequences=True),
    SimpleRNN(50, return_sequences=True),
    SimpleRNN(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

import time
start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)
end = time.time()
print(f"Training time: {end - start} seconds")


#LSTM MODEL
#LSTM MODEL
# Create the LSTM model
lst_model = Sequential()
lst_model.add(LSTM(units=200,  return_sequences=True,input_shape=(seq_length, 1)))
lst_model.add(LSTM(units=100 , return_sequences=True ))
lst_model.add(LSTM(units=50 , return_sequences=False ))
lst_model.add(Dense(units=1))

# Compile the model
lst_model.compile(optimizer='adam', loss='mean_squared_error')

 # Train the model on the training set
start = time.time()
history = lst_model.fit(X_test, y_test,validation_data=(X_val, y_val),epochs=20, batch_size=50, verbose=1)
end = time.time()
print(f"Training time: {end - start} seconds")