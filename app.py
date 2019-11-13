import requests,csv,sqlite3
import csv
#import datetime
#import json
#import sys
import traceback
import urllib.request
from pathlib import Path
import settings
#import pandas as pd

#For Arima
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

from matplotlib.figure import Figure

#For Regression
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

#Forother
from apscheduler.scheduler import Scheduler

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'qwertyuiopsecretkey'

db = SQLAlchemy(app)


class City(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)


# app.config['DEBUG'] = True

key = 'c9f1f34212b6ce232a81ca55ffc01e4f'
unit = 'metric'
url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units={}&appid={}'


def get_weather_data(city):
    r = requests.get(url.format(city, unit, key)).json()
    return r


def update_scheduler():
    print('Updating.....')
    update()

sched = Scheduler()
sched.add_interval_job(test_scheduler, days=1)
sched.start()

@app.route('/current')
def index_get():
    cities = City.query.all()
    
    weather_data = []
    
    
    for city in cities:
        r = get_weather_data(city.name)
        # print(r)
        #
        weather = {
            'city': city.name,
            'temperature': round(r['main']['temp']),
            'description': str(r['weather'][0]['description']).title(),
            'icon': r['weather'][0]['icon'],
            'humidity':r['main']['humidity'],
            'wind': r['wind']['speed']
        }
        
        weather_data.append(weather)
        
        
    
    return render_template('current.html', weather_data=weather_data)


@app.route('/current', methods=['POST'])
def index_post():
    err_msg = ''
    
    new_city = request.form.get('city').strip().lower()
    
    res = get_weather_data(new_city)
    
    if res['cod'] == 200 and new_city:
        
        city_exists = City.query.filter(City.name.ilike(f"%{new_city}%")).first()
        
        if not city_exists:
            
            city_to_add = City(name=new_city.title())
            db.session.add(city_to_add)
            db.session.commit()
        else:
            err_msg = 'City Already Exists'
    else:
        err_msg = 'City does not exist in the world !'
    
    if err_msg:
        flash(err_msg, 'error')
    else:
        flash('City Added Successfully !', 'success')
    
    return redirect(url_for('index_get'))

@app.route('/historic',methods = ['POST','GET'])

def historic():
    if request.method == 'POST':
        city = request.form['city']
        print(city)
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        if not end_date:
            end_date=float("inf")
        con = sqlite3.connect('weather.db')
        con.row_factory = sqlite3.Row

        cur = con.cursor()
        command="select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data where time>\""+str(start_date)+"\" and time<=\""+str(end_date)+"\""

        if request.form.get('citycheck'):
            command+=" and city='"+str(city)+"'"
        if request.form.get('temperatureMin'):
            command+=" and temperatureMin"+request.form['minTempCompare']+""+request.form['minTempVal']+""
        if request.form.get('temperatureMax'):
            command+=" and temperatureMax"+request.form['maxTempCompare']+""+request.form['maxTempVal']+""
        if request.form.get('precipIntensity'):
            command+=" and precipIntensity"+request.form['precipIntensityCompare']+""+request.form['precipIntensityVal']+""
        command+=" order by time desc"
        print(command)
        cur.execute(command)
        search = cur.fetchall();
        
        cur.execute("select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data")
        rows = cur.fetchall();
        
        cur.execute("select * from locations")
        list_of_locations = cur.fetchall()

        return render_template('historic.html',rows=rows,search=search,list_of_locations=list_of_locations)
    else:
        #return render_template("historic.html")
        con = sqlite3.connect('weather.db')
        con.row_factory = sqlite3.Row

        cur = con.cursor()
        cur.execute("select * from locations")

        list_of_locations = cur.fetchall();
        return render_template("historic.html",list_of_locations = list_of_locations)

@app.route('/')
def main():
    return render_template('info.html')

@app.route('/update')
def update():

    def get_url(day_location):
        day = '{:{dfmt}}'.format(day_location[0], dfmt='%Y-%m-%d')
        location = str(day_location[1][1]) + ',' + str(day_location[1][2])

        return """https://api.darksky.net/forecast/{ACCESS_TOKEN}/{location},{date}T23:59:59?units=si""".format(
            location=location, date=day, ACCESS_TOKEN=settings.DARKSKY_ACCESS_TOKEN)


    def str_time(unix_time):
        if unix_time is None:
            return None
        else:
            return datetime.datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')


    def readcsv(file_name):
        weather_file = Path(file_name)
        if weather_file.is_file():
            with open(weather_file, newline='') as f:
                return [row for row in csv.DictReader(f)]
        else:
            return []


    def writecsv(file_name, header, weather_history):
        with open(file_name, 'w', newline='') as fp:
            writer = csv.DictWriter(fp, delimiter=',', fieldnames=header)
            writer.writeheader()
            writer.writerows(weather_history)


    def get_existing_dates_and_locations(weather_history):
        existing_days_and_locations = set()
        for x in weather_history:
            daytime = datetime.datetime.strptime(x["time"], "%Y-%m-%d %H:%M:%S")
            day = daytime.date()
            location = (x["city"], float(x["latitude"]), float(x["longitude"]))
            existing_days_and_locations.add((day, location))
        return existing_days_and_locations


    required_fields = [
        "time", "timezone", "latitude", "longitude", "summary", "sunriseTime", "sunsetTime",
        "precipIntensity", "precipIntensityMax", "precipIntensityMaxTime",
        "precipProbability", "precipType",
        "temperatureMin", "temperatureMinTime", "temperatureMax", "temperatureMaxTime",
        "apparentTemperatureMin", "apparentTemperatureMinTime", "apparentTemperatureMax",
        "apparentTemperatureMaxTime"
    ]


    def get_expected_dates_and_locations(days_back, locations):
        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=days_back)
        step = datetime.timedelta(days=1)
        expected_days_and_locations = set()
        while end > start:
            for l in locations:
                expected_days_and_locations.add((end.date(), l))
            end -= step
        return expected_days_and_locations


    def get_weather_data(dates_and_locations):
        weather_history = []
        for day_location in dates_and_locations:
            url = get_url(day_location)
            print('getting data from {}'.format(url))
            try:
                raw_data = json.loads(urllib.request.urlopen(url).read())
                one_day_data = {key: value for key, value in raw_data["daily"]["data"][0].items() if key in required_fields}
                for required_field in required_fields:
                    if required_field not in one_day_data:
                        one_day_data[required_field] = None

                daylight = str((datetime.datetime.fromtimestamp(one_day_data["sunsetTime"])) - (
                    datetime.datetime.fromtimestamp(one_day_data["sunriseTime"])))
                one_day_data['daylight'] = daylight
                one_day_data['timezone'] = raw_data["timezone"]
                one_day_data['city'] = day_location[1][0]
                one_day_data['latitude'] = day_location[1][1]
                one_day_data['longitude'] = day_location[1][2]
                one_day_data['time'] = str_time(one_day_data['time'])
                one_day_data['sunriseTime'] = str_time(one_day_data['sunriseTime'])
                one_day_data['sunsetTime'] = str_time(one_day_data['sunsetTime'])
                one_day_data['temperatureMinTime'] = str_time(one_day_data['temperatureMinTime'])
                one_day_data['apparentTemperatureMinTime'] = str_time(one_day_data['apparentTemperatureMinTime'])
                one_day_data['apparentTemperatureMaxTime'] = str_time(one_day_data['apparentTemperatureMaxTime'])
                one_day_data['precipIntensityMaxTime'] = str_time(one_day_data['precipIntensityMaxTime'])
                one_day_data['temperatureMaxTime'] = str_time(one_day_data['temperatureMaxTime'])

                weather_history.append(one_day_data)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Missing data in " + str(day_location))
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)

        return weather_history


    existing_data = readcsv(settings.WEATHER_HISTORY_FILE)
    existing_days_and_locations = get_existing_dates_and_locations(existing_data)

    con = sqlite3.connect("weather.db")
    cur = con.cursor()
    cur.execute("select * from locations")
    listOfLocations_db=set(["a", "b","c"])
    listOfLocations_db.clear()
    while True:
        row = cur.fetchone()
        if row == None:
            break
        listOfLocations_db.add(row)
    print("LIST OF LOCATIONS",listOfLocations_db)
    #print("SETTINGS",settings.LOCATIONS1)

    con.close()
   

    expected_days_and_locations = get_expected_dates_and_locations(settings.DAYS_BACK, listOfLocations_db)
    #expected_days_and_locations = get_expected_dates_and_locations(settings.DAYS_BACK, settings.LOCATIONS)
    missing_days_and_locations = expected_days_and_locations - existing_days_and_locations
    missing_data = get_weather_data(missing_days_and_locations)
    writecsv(settings.WEATHER_HISTORY_FILE, required_fields + ['daylight'] + ['city'], existing_data + missing_data)

    conn = sqlite3.connect('weather.db')
    c = conn.cursor()

    df = pd.read_csv('weather.csv')
    df.to_sql('weather_data', conn, if_exists='replace' , index=False)

    return render_template('update.html')

@app.route('/locations',methods = ['POST','GET'])
def locations_from_db():
    conn = sqlite3.connect('weather.db')
    print("Opened database successfully")
    conn.execute('CREATE TABLE IF NOT EXISTS locations (place TEXT, latitude INTEGER, longitude INTEGER)')

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select * from locations")
    list_of_locations = cur.fetchall();
    if request.method == 'POST':
        place = request.form['place']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        
        if request.form['btn'] =='add':
            cur.execute("INSERT INTO locations values(?,?,?)",(place,latitude,longitude))
            conn.commit()
        elif request.form['btn'] =='delete':
            cur.execute("DELETE FROM locations where place=? and latitude=? and longitude=?",(place,latitude,longitude))
            conn.commit()

        cur.execute("select * from locations")
        list_of_locations = cur.fetchall();

    conn.close()
    return render_template('locations.html',list_of_locations=list_of_locations)
    

@app.route('/prediction',methods = ['POST','GET'])
def prediction():
    conn = sqlite3.connect('weather.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select * from locations order by place")
    list_of_locations = cur.fetchall();
    
    if request.method == 'POST':
        city = request.form['city']
        feature_to_predict=['temperatureMin','temperatureMax','precipIntensity']
        method = request.form['btn']
        if method == 'arima' :
            finalresults=[]
            df = pd.read_csv("weather.csv").set_index("time")
            for predictfeature in feature_to_predict:
                df = df.loc[df['city'] == city]

                df.index = pd.to_datetime(df.index)
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
                    t= datetime.datetime(int(t[0]),int(t[1]),int(t[2])) 
                    lastdate = t

                #y.plot(figsize=(15, 6))


                p = d = q = range(0, 2)

                # Generate all different combinations of p, q and q triplets
                pdq = list(itertools.product(p, d, q))
                #print(type(itertools.product(p, d, q)))
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
                #print(pred_uc.predicted_mean.tolist())
                test=[predictfeature]
                test.append(pred_uc.predicted_mean.tolist())
                finalresults.append(test)
                #finalresults=[[18.210447161876697, 17.627294675072463, 17.772584811128002, 17.02610372085857, 16.136252754772723, 16.171641735303865, 17.152165177023615], 'temperatureMin', [28.86706957612906, 29.415799800772703, 29.064417911082437, 24.633105519904852, 21.93034457593456, 22.579520798401838, 26.067165441560277], 'temperatureMax', [-0.02904041300196783, -0.08500225546115066, -0.025428453189474773, 0.16945112701774584, 0.20177596401609943, 0.37224683674247017, -0.0959111490897569], 'precipIntensity']
                print(test)
                print(finalresults)
                # Get confidence intervals of forecasts
                pred_ci = pred_uc.conf_int()
                plt.clf() 
                ax = y.plot(label='observed', figsize=(20, 15))
                pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
                ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0],
                                pred_ci.iloc[:, 1], color='k', alpha=.25)
                ax.set_xlabel('Date')
                ax.set_ylabel(predictfeature)

                plt.legend()

                if predictfeature == 'precipIntensity':
                    loc3 = "static/graphs/" + predictfeature + ".png"
                    plt.savefig(loc3) 
                    plt.clf() 
                elif predictfeature == 'temperatureMin':
                    loc1 = "static/graphs/" + predictfeature + ".png"
                    plt.savefig(loc1)
                    plt.clf() 
                elif predictfeature == 'temperatureMax':
                    loc2 = "static/graphs/" + predictfeature +  ".png"
                    plt.savefig(loc2)
                    plt.clf() 
                else:
                    print("Invalid feature")
                    exit(1)
            
            return render_template('prediction.html',list_of_locations=list_of_locations,method='arima',results=finalresults,url1 =loc1, url2= loc2 , url3=loc3 )


        if method == 'regression' :
            city = request.form['city']
            feature_to_predict=['temperatureMin','temperatureMax','precipIntensity']
            results=[]
            for predictfeature in feature_to_predict:
               
                def get_target_date():
                    """Return target date 1000 days prior to current date."""
                    current_date = datetime.now()
                    target_date = current_date - timedelta(days=1000)
                    return target_date

                def derive_nth_day_feature(df, feature, N):
                    nth_prior_measurements = df[feature].shift(periods=N)
                    col_name = f'{feature}_{N}'
                    df[col_name] = nth_prior_measurements

                features = [
                        'time', 'precipIntensity', 'precipIntensityMax',
                        'precipProbability',
                       'temperatureMin',  'temperatureMax',
                        'apparentTemperatureMin',
                        'apparentTemperatureMax',
                ]

                df = pd.read_csv("weather.csv").set_index("time")

                df = df.loc[df['city'] == city]
                df.dropna()

                print(df.sort_values("time"))
                print(df.columns)

                nextday = datetime.datetime.today() 
                nextday += datetime.timedelta(days=1)
                temp = str(nextday).split(" ")[0]
                temp = (temp).split("-") 
                temp = datetime.datetime(int(temp[0]),int(temp[1]),int(temp[2]))

                nextday = temp
                print(nextday)

                record = [[nextday,'','','','','','','']]
                newdf = pd.DataFrame(record, columns=features).set_index('time')
                print(newdf)

                df.index = pd.to_datetime(df.index)
                newdf.index = pd.to_datetime(newdf.index)

                features = [
                        'precipIntensity', 'precipIntensityMax',
                        'precipProbability',
                       'temperatureMin',  'temperatureMax',
                        'apparentTemperatureMin',
                        'apparentTemperatureMax',
                ]

                data = df[features]

                data = data.sort_values(by=['time'])
                data = data.resample('d').mean().dropna(how='all')
                #print("Edited database with no dublicates \n", data)
                data = data.append(newdf)

                df = data

                # target measurement of mean temperature
                ft = [predictfeature]

                #print(tmp[feature][1])
                # a list representing Nth prior measurements of feature
                # notice that the front of the list needs to be padded with N
                # None values to maintain the constistent rows length for each N

                for feature in features:
                    if feature != 'time':
                            for N in range(1, 4):
                               derive_nth_day_feature(df, feature, N)

                print("Dataframe with nth day features: " , df)

                to_remove = [
                    feature for feature in features
                    if feature not in ft
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

                df_corr = df.corr()[[predictfeature]].sort_values(predictfeature)
                #print(df_corr)   
                df_corr_fil = df_corr[abs(df_corr[predictfeature]) > 0.30]
                #print(df_corr_fil)

                unwanted = [predictfeature]
                predictors = df_corr_fil.index.tolist()


                predictors = [i for i in predictors if i not in unwanted]
                print("Predictors: ", predictors)

                df2 = df[[predictfeature] + predictors]
                trial = trial[[predictfeature] + predictors]

                X = df2[predictors]
                trial = trial[predictors]

                y = df2[predictfeature]
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
                    
                    included = initial_list
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

                values=[predictfeature,predicttest[0],float(regressor.score(X_test, y_test)),float(mean_absolute_error(
                    y_test, prediction)),median_absolute_error(y_test, prediction)]
                results.append(values)


            return render_template('prediction.html',list_of_locations=list_of_locations,method='regression',results=results)

        if method == 'neuralnetwork' :
            pass


    return render_template('prediction.html',list_of_locations=list_of_locations)

@app.route('/list')
def listview():
   con = sqlite3.connect('weather.db')
   con.row_factory = sqlite3.Row

   cur = con.cursor()
   cur.execute("select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data")
        
   rows = cur.fetchall();
   print(rows)
   return render_template("list.html",rows = rows)

@app.route('/delete/<name>')
def delete_city(name):
    city = City.query.filter_by(name=name).first()
    db.session.delete(city)
    db.session.commit()
    
    flash('Successfully Deleted {}'.format(city.name),'success')
    
    return redirect(url_for('index_get'))


if __name__ == "__main__":
    app.run(debug=True)
