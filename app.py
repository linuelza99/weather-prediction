import requests,csv,sqlite3
import csv
import datetime
import json
import sys
import traceback
import urllib.request
from pathlib import Path
import settings
import pandas


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
        """if (city!='All'):
            cur.execute("select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data where city=? and time>? and time<=?",(city,start_date,end_date))
        else:
            cur.execute("select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data where time>? and time<=?",(start_date,end_date))
        """
        #cur.execute("select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data where time>? and time<=?",(start_date,end_date))
        cur.execute(command)
        
        
        search = cur.fetchall();
        cur.execute("select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data")
        rows = cur.fetchall();


        return render_template('historic.html',rows=rows,search=search)
    else:
        return render_template("historic.html")
        con = sqlite3.connect('weather.db')
        con.row_factory = sqlite3.Row

        cur = con.cursor()
        cur.execute("select city,time,summary,temperatureMin,temperatureMax,precipIntensity,precipProbability,precipType,daylight from weather_data")

        rows = cur.fetchall();
        print(rows)
        return render_template("list.html",rows = rows)

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
    expected_days_and_locations = get_expected_dates_and_locations(settings.DAYS_BACK, settings.LOCATIONS)
    missing_days_and_locations = expected_days_and_locations - existing_days_and_locations
    missing_data = get_weather_data(missing_days_and_locations)
    writecsv(settings.WEATHER_HISTORY_FILE, required_fields + ['daylight'] + ['city'], existing_data + missing_data)

    conn = sqlite3.connect('/home/linu/GIT repos/weather-prediction/weather.db')
    c = conn.cursor()

    df = pandas.read_csv('weather.csv')
    df.to_sql('weather_data', conn, if_exists='replace' , index=False)

    return render_template('update.html')



@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/list')
def list():
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
