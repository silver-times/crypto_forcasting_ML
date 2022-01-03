import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.tools.datetimes import Scalar 
import pandas_datareader as web
import datetime as dt
import yfinance as yfin
yfin.pdr_override()

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential 


crypto_currency = "BTC"
against_currency = "USD"

data = web.get_data_yahoo(f'{crypto_currency}-{against_currency}', start="2018-01-01", end="2021-11-01")


#---------------------------------
#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60
# future_day = 30

x_train, y_train = [], [] 

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



#---------------------------------
#Create Neural Network
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size = 32)


#---------------------------------
#Testing the Model

test_data = web.get_data_yahoo(f'{crypto_currency}-{against_currency}', start="2020-01-01", end="2021-11-01")
actual_prices = test_data['Close'].values


total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)


# x_test = []

# for x in range(prediction_days, len(model_inputs)):
#     x_test.append(model_inputs[x-prediction_days:x, 0])

# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# prediction_prices = model.predict(x_test)
# prediction_prices = scaler.inverse_transform(prediction_prices)

# plt.plot(actual_prices, color='crimson', label='Actual Prices')
# plt.plot(prediction_prices, color='teal', label='Predicted Prices')
# plt.title(f'{crypto_currency} price prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend(loc='upper left')
# plt.show()



#---------------------------------
#Predict Next Day
real_data = []

for x in range(prediction_days, len(model_inputs) + 1):
    real_data.append(model_inputs[x-prediction_days:x, 0])

# real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
# real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs) + 1, 0]]


real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))


prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

plt.plot(actual_prices, color='crimson', label='Actual Prices')
plt.plot(prediction, color='yellowgreen', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
