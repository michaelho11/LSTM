import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

df = yf.download('0700.HK', start = '2022-01-01')

df = df.tail(10)

scaler = MinMaxScaler(feature_range = (0,1))
scaled_df = scaler.fit_transform(df)

print (scaled_df)