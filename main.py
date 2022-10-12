import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

#download dataframe
df = yf.download('0700.HK', start = '2022-01-01')

df = df.head(6)

# normalised the dataframe for sake of easy comparison
scaler = MinMaxScaler(feature_range = (0,1))
scaled_df = scaler.fit_transform(df)
print (df.to_string())
print (scaled_df)

print ('separation line 1')

# data engineering
# for y_train we choose OPEN hence
# y_train [:,0] to take all the first item (open) in every list
backward_counting = 3
x_train = np.array([scaled_df[i:i+backward_counting] for i in range(len(scaled_df - backward_counting))], dtype = object)
y_train = np.array([scaled_df[:,0][i:i+backward_counting] for i in range(len(scaled_df - backward_counting))], dtype = object)

print (x_train)

print ('separation line 2')

print (y_train)






