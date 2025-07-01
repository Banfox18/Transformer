import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

ticker = 'daily' 
data = pd.read_csv('Data/CSI 300/' + ticker + '.csv')
openPrice = data['OpenPrice']
highPrice = data['HighPrice']
lowPrice = data['LowPrice']
closePrice = data['ClosePrice']
volume = data['Volume']
changeRatio = data['ChangeRatio']
data['Date'] = pd.to_datetime(data['CloseDate'])

features = np.stack([openPrice, highPrice, lowPrice, closePrice, volume, changeRatio], axis=1)

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

window_size = 10
X, y = [], []
for i in range(len(features_scaled) - window_size):
    X.append(features_scaled[i:i+window_size])
    y.append(features_scaled[i+window_size][3])  # 第4列为收盘价

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.LSTM(50, return_sequences=False, input_shape=(window_size, X.shape[2])),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test), verbose=2)
close_scaler = MinMaxScaler()
close_scaler.fit(closePrice.values.reshape(-1,1))

y_pred = model.predict(X_test)
y_pred_real = close_scaler.inverse_transform(y_pred)
y_test_real = close_scaler.inverse_transform(y_test.reshape(-1,1))
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print(f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}")