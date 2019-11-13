import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import requests
import json
import sys
import os
import pyodbc
import datetime

import pandas as pd

import collections
import time
from datetime import datetime, timedelta
import os

df = pd.read_csv("weather.csv").set_index("time")

city = str(input("Enter city: "))
#city = "Karnataka"
df = df.loc[df['city'] == city]

df.index = pd.to_datetime(df.index)
#df.sort_values("time")
predictfeature = str(input("Enter feature to predict: "))
#predictfeature = "temperatureMax"
feature = [predictfeature]

data = df[feature]
print(data)
y = data

# The 'MS' string groups the data in buckets by start of the month

if predictfeature == 'precipIntensity':
    y = y.precipIntensity.resample('d').mean()
elif predictfeature == 'precipIntensityMax':
    y = y.precipIntensityMax.resample('d').mean()
elif predictfeature == 'precipProbability':
    y = y.precipProbability.resample('d').mean()
elif predictfeature == 'temperatureMin':
    y = y.temperatureMin.resample('d').mean()
elif predictfeature == 'temperatureMax':
    y = y.temperatureMax.resample('d').mean()
else:
    print("Invalid feature")
    exit(1)  

# The term bfill means that we use the value before filling in missing values
y = y.fillna(y.bfill())
print(y)

temp = y.head(1)
print(temp)
temp = np.array(temp.index)
print("temp = " , temp)

lastdate = ''
for i in  temp:
    t = str(i).split("T")
    t = t[0]
    t = t.split("-") 
    t= datetime(int(t[0]),int(t[1]),int(t[2])) 
    lastdate = t

y.plot(figsize=(15, 6))
plt.show()

p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#print('Examples of parameter combinations for Seasonal ARIMA...')
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=False)

results = mod.fit()




pred = results.get_prediction(start=pd.to_datetime(lastdate), dynamic=False)
pred_ci = pred.conf_int()
print(pred.predicted_mean)


y_forecasted = pred.predicted_mean
y_truth = y[lastdate:]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our dynamic forecasts is {}'.format(round(mse, 2)))

pred_dynamic = results.get_prediction(start=pd.to_datetime(lastdate), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

#ax = y['2017':].plot(label='observed', figsize=(20, 15))
#pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
print(pred_dynamic.predicted_mean)



# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y[lastdate:]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=7)
print(pred_uc.predicted_mean)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel(predictfeature)

plt.legend()
plt.show()

