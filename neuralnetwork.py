import numpy as np
import tensorflow as tf
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             median_absolute_error)
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
"""
required_fields = [
    "time", "latitude", "longitude" , "precipIntensity","precipIntensityMax","precipProbability","dewPoint",
    "humidity","pressure","windSpeed","cloudCover","visibility","temperatureMin","temperatureMax"
]
"""

required_fields = [
        'time' , 'precipIntensity', 'precipIntensityMax',
        'precipProbability',
       'temperatureMin',  'temperatureMax',
        'apparentTemperatureMin',
        'apparentTemperatureMax'
]

features = [
        'time' , 'latitude' , 'longitude' ,'precipIntensity', 'precipIntensityMax',
        'precipProbability',
       'temperatureMin',  'temperatureMax',
        'apparentTemperatureMin',
        'apparentTemperatureMax'
]

def derive_nth_day_feature(df, feature, N):
    nth_prior_measurements = df[feature].shift(periods=N)
    col_name = f'{feature}_{N}'
    df[col_name] = nth_prior_measurements


df = pd.read_csv("weather.csv").set_index("time")
df.index = pd.to_datetime(df.index)

df = df.loc[df['city'] == "Mangalore"]

df = df.sort_values(by=['time'])
df = df.resample('d').mean().dropna(how='all')

data = df.copy()

for feature in required_fields:
    if feature == "longitude":
        print("skip")
    elif feature == "latitude":
        print("skip")
    elif feature != 'time':
            for N in range(1, 4):
               derive_nth_day_feature(df, feature, N)


to_remove = [
    feature for feature in required_fields
    if feature not in ['temperatureMin', 'temperatureMax']
]
#print(to_remove)

# make a list of columns to keep
to_keep = [col for col in df.columns if col not in to_remove]
#print(to_keep)

# select only the columns in to_keep and assign to df
df = df[to_keep]
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()    
 
#print(df.columns)

#print(df.describe().T)

df.reset_index()
df.index = df.index.values.astype(float)



# First drop the maxtempm and mintempm from the dataframe
df = df.drop(['temperatureMax'], axis=1)
df = df.drop(['latitude'], axis=1)
df = df.drop(['longitude'], axis=1)
print(df)   

# X will be a pandas dataframe of all columns except meantempm
X = df[[col for col in df.columns if col != 'temperatureMin']]

# y will be a pandas series of the meantempm
y = df['temperatureMin']
print("Dataframe X \n" , X)
print("Dataframe Y \n" , y)



# split data into training set and a temporary set
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, random_state=23)

# split the remaining 20% of data evenly
X_test, X_val, y_test, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=23)

X_train.shape, X_test.shape, X_val.shape
print('Training instances   {}, Training features   {}'.format(
    X_train.shape[0], X_train.shape[1]))
print('Validation instances {}, Validation features {}'.format(
    X_val.shape[0], X_val.shape[1]))
print('Testing instances    {}, Testing features    {}'.format(
    X_test.shape[0], X_test.shape[1]))

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]
#print(feature_cols)

regressor = tf.estimator.DNNRegressor(
    feature_columns=feature_cols,
    hidden_units=[50, 50],
    model_dir='tf_wx_TempMin'
    )


def wx_input_fn(X, y=None, num_epochs=None, shuffle=False, batch_size=400):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=X,
        y=y,
        num_epochs=num_epochs,
        shuffle=shuffle,
        batch_size=batch_size)


evaluations = []
STEPS = 100
for i in range(50):    
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(
        regressor.evaluate(
            input_fn=wx_input_fn(X_val, y_val, num_epochs=1, shuffle=False)))

pred = regressor.predict(
    input_fn=wx_input_fn(X_test, num_epochs=1, shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])

print('The Explained Variance: %.2f' % explained_variance_score(
    y_test, predictions))
print('The Mean Absolute Error: %.2f degrees Celcius' % mean_absolute_error(
    y_test, predictions))
print('The Median Absolute Error: %.2f degrees Celcius' %
      median_absolute_error(y_test, predictions))


data = data.tail(3)

nextday = datetime.datetime.today() 

temp = str(nextday).split(" ")[0]
temp = (temp).split("-") 
#print(temp)
temp = datetime.datetime(int(temp[0]),int(temp[1]),int(temp[2]))
#print(temp)

nextday = temp
print(nextday)

record = [[nextday,'','','','','','','','','']]
newdf = pd.DataFrame(record, columns=features).set_index('time')
print(newdf)
data = data.append(newdf)



for feature in required_fields:
    if feature == "longitude":
        print("skip")
    elif feature == "latitude":
        print("skip")
    elif feature != 'time':
            for N in range(1, 4):
               derive_nth_day_feature(data, feature, N)

data = data[to_keep]
print(data)

data = data.drop(['temperatureMax'], axis=1)
data = data.drop(['temperatureMin'], axis=1)
data = data.drop(['latitude'], axis=1)
data = data.drop(['longitude'], axis=1)

data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()  

data.reset_index()
data.index = data.index.values.astype(float)

print(data)
pred = regressor.predict(
    input_fn=wx_input_fn(data, num_epochs=1, shuffle=False))

predictiontest = np.array([p['predictions'][0] for p in pred])
print("Prediction for next day is:", predictiontest)