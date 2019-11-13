# Python program to find current 
# weather details of any city 
# using openweathermap api 

# import required modules 

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split

import requests, json 
import collections
import time
import datetime
import os


def get_target_date():
    """Return target date 1000 days prior to current date."""
    current_date = datetime.now()
    target_date = current_date - timedelta(days=1000)
    return target_date

def derive_nth_day_feature(df, feature, N):
    nth_prior_measurements = df[feature].shift(periods=N)
    col_name = f'{feature}_{N}'
    df[col_name] = nth_prior_measurements


"""
features = [
    "time", "precipIntensity","precipIntensityMax","precipProbability","dewPoint",
    "humidity","pressure","windSpeed","cloudCover","visibility","temperatureMin","temperatureMax"]
    """


features = [
        'time', 'precipIntensity', 'precipIntensityMax',
        'precipProbability',
       'temperatureMin',  'temperatureMax',
        'apparentTemperatureMin',
        'apparentTemperatureMax',
]

df = pd.read_csv("weather.csv").set_index("time")
df = df.loc[df['city'] == "Mangalore"]

print(df.sort_values("time"))
print(df.columns)



nextday = datetime.datetime.today() 

temp = str(nextday).split(" ")[0]
temp = (temp).split("-") 
#print(temp)
temp = datetime.datetime(int(temp[0]),int(temp[1]),int(temp[2]))
#print(temp)

nextday = temp
print(nextday)

record = [[nextday,'','','','','','','']]
newdf = pd.DataFrame(record, columns=features).set_index('time')
print(newdf)

df.index = pd.to_datetime(df.index)
newdf.index = pd.to_datetime(newdf.index)
#print(df.sort_values)
#print(df.columns)

"""
features = [
    "precipIntensity","precipIntensityMax","precipProbability","dewPoint",
    "humidity","pressure","windSpeed","cloudCover","visibility","temperatureMin","temperatureMax"
]
"""
features = [
        'precipIntensity', 'precipIntensityMax',
        'precipProbability',
       'temperatureMin',  'temperatureMax',
        'apparentTemperatureMin',
        'apparentTemperatureMax',
]

data = df[features]
#print(data)    


data = data.sort_values(by=['time'])
data = data.resample('d').mean().dropna(how='all')
#print("Edited database with no dublicates \n", data)
data = data.append(newdf)

df = data

#print(data.loc[nextday])

#print("Database: ", df)

#tmp = df[['temperatureMean', 'dewPoint']].head(4)
#print(tmp)
# target measurement of mean temperature
feature = 'temperatureMin'

# total number of rows
#rows = tmp.shape[0]

#print(tmp[feature][1])
# a list representing Nth prior measurements of feature
# notice that the front of the list needs to be padded with N
# None values to maintain the constistent rows length for each N

for feature in features:
    if feature != 'time':
	        for N in range(1, 4):
 	           derive_nth_day_feature(df, feature, N)

#print("Dataframe with nth day features: " , df)

to_remove = [
    feature for feature in features
    if feature not in ['temperatureMin', 'temperatureMax']
]
#print(to_remove)

# make a list of columns to keep
to_keep = [col for col in df.columns if col not in to_remove]
#print(to_keep)

# select only the columns in to_keep and assign to df
df = df[to_keep]

df = df.apply(pd.to_numeric, errors='coerce')

#print(df.info())
# Call describe on df and transpose it due to the large number of columns
spread = df.describe().T

# precalculate interquartile range for ease of use in next calculation
IQR = spread['75%'] - spread['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile	
spread['outliers'] = (spread['min'] <(spread['25%'] -(3 * IQR))) | (spread['max'] > (spread['75%'] + 3 * IQR))
#print(spread)
#print(spread.iloc[spread.outliers,])

#print("Current: ",df)

trial = df.loc[nextday]
#print("Testing dataset: " , trial)

df = df.dropna()
#print(df)

df_corr = df.corr()[['temperatureMin']].sort_values('temperatureMin')
#print(df_corr)   
df_corr_fil = df_corr[abs(df_corr['temperatureMin']) > 0.40]
#print(df_corr_fil)


unwanted = ['temperatureMin', 'temperatureMax']
predictors = df_corr_fil.index.tolist()
predictors = [i for i in predictors if i not in unwanted]
#print(predictors)

df2 = df[['temperatureMin'] + predictors]
trial = trial[['temperatureMin'] + predictors]


X = df2[predictors]
trial = trial[predictors]

y = df2['temperatureMin']
alpha = 0.05

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)
#print("Testing dataset: ", trial)
#print("X dataset: ", X)



def stepwise_selection(X,
                       y,
                       initial_list=predictors,
                       threshold_out=alpha,
                       verbose=True):
    
    included = list(initial_list)
    #print("Initial list : ", initial_list)
    

    while True:
    	#print("List:", included)
    	changed = False
    	model = sm.OLS(y,X[included]).fit()
    	# use all coefs except intercept
    	pvalues = model.pvalues.iloc[1:]
    	#print("Values: ", pvalues)
    	worst_pval = pvalues.max()  # null if pvalues is empty
    	if worst_pval > threshold_out:
    		changed = True
    		worst_feature = pvalues.idxmax()
    		#print("Worst Feature:", worst_feature)
    		included.remove(worst_feature)
    		#print("List:", included)
    		if verbose:
    			print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
    	if not changed:
    		break
    return included


result = stepwise_selection(X, y)

print('Resulting features:')
print(result)

X = X[result]
trial=trial[result]
#print("X: ", X)
#print("Testing: ", trial)

model = sm.OLS(y, X).fit()
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

prediction = regressor.predict(X_test)
#print("X_test : " , X_test)

#print("Prediction: ", prediction)


trial = [trial]
print(trial)
predicttest = regressor.predict(trial)
print("Prediction of testing: ", predicttest)


print('The Explained Variance: %.2f' % regressor.score(X_test, y_test))
print('The Mean Absolute Error: %.2f degrees celcius' % mean_absolute_error(
    y_test, prediction))
print('The Median Absolute Error: %.2f degrees celcius' %
      median_absolute_error(y_test, prediction))
