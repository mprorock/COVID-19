#!/usr/bin/env python
# coding: utf-8
# run initial imports
from __future__ import print_function
import warnings
#warnings.filterwarnings('ignore')
import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go

from scipy.integrate import odeint
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import widgets

import folium
from folium.plugins import FastMarkerCluster

from datetime import datetime
import os


NOW = datetime.now().strftime("%Y/%m/%d %H:%M")
NOW_FILE = datetime.now().strftime("%Y_%m_%d__%H%M")
pDay=3 # number of corecast days - should not exceed 5 unless you are very careful in tuning, and understand how to interpert the results
input_dir = './data/'
covid_19_national_observations = pd.read_csv(input_dir + 'global/covid_19_national_observations.csv')
covid_19_infected_observations = pd.read_csv(input_dir + 'global/covid_19_infected_observations.csv')

countries = covid_19_national_observations['Country/Region'].unique()
countries.sort()

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def cleanStr(s):
    c = s.lower().replace(' ', '_')
    c = c.replace('*', '').replace('(', '_').replace(')', '_').replace(',', '_')
    c = c.strip()
    return c

def getDf(country):
    c = cleanStr(country)
    #setup initial data frames
    input_dir = './data/countries/' + c + '/'
    cases = pd.read_csv(input_dir + 'covid_19_'+ c +'_cases.csv')
    totals = pd.read_csv(input_dir + './covid_19_' + c + '_totals.csv')
    
    #correct date parsing on some of the JH data
    cases['Date'] = pd.to_datetime(cases['Date'])
    totals['Date'] = pd.to_datetime(totals['Date'])
    
    cases = cases.sort_values('Date')
    totals = totals.sort_values('Date')
    
    cases = cases[cases['Confirmed'] > 0]
    return cases, totals

def pred_country(country, t=90, infectivity_factor=180, gMethod='linear', disp=True):
    cases, totals = getDf(country)
    
    # now let's run the forecast with fbprophet
    fb_df = totals[['Date', 'Active Cases']].copy()
    fb_df = fb_df.sort_values('Date').reset_index(drop=True)
    fb_df.columns = ['ds','y']
    #fb_df['cap'] = totals['Death'] * infectivity_factor
    fb_df['floor'] = 0
    #print(fb_df)

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, growth=gMethod)
    m.fit(fb_df)
    future = m.make_future_dataframe(periods=t)
    #future['cap'] = totals['Death'].max() * infectivity_factor
    future['floor'] = 0
    forecast = m.predict(future)
    py.init_notebook_mode()

    fig = plot_plotly(m, forecast, xlabel='Date', ylabel='Active Cases', uncertainty=True, figsize=(1100,600))  # This returns a plotly Figure
    fig.update_layout(title='Active '+country+' COVID-19 Cases and Forecast ('+str(t)+' day) as of' + str(NOW))
    
    c = cleanStr(country)
    os.makedirs("./images/" + c + "/" + NOW_FILE, exist_ok=True)
    fig.write_image("./images/" + c + "/" + NOW_FILE + "/" + NOW_FILE + "__" + c + "_forecast_" + str(t) + "_day.png")
    
    if disp:
        py.iplot(fig)

def pred_province(country, province, t=90, infectivity_factor=180, gMethod='linear', disp=True):
    cases, totals = getDf(country)
    if len(cases['Province/State'].unique()) > 1:
        cases = cases[cases['Province/State'] == province]
    
    #print('Total records so far: ', totals.shape[0])
    # now let's run the forecast with fbprophet
    fb_df = cases[['Date', 'Confirmed']].copy()
    fb_df = fb_df.sort_values('Date').reset_index(drop=True)
    fb_df.columns = ['ds','y']
    #fb_df['cap'] = totals['Death'] * infectivity_factor
    fb_df['floor'] = 0
    #print(fb_df)

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, growth=gMethod)
    m.fit(fb_df)
    future = m.make_future_dataframe(periods=t)
    #future['cap'] = totals['Death'].max() * infectivity_factor
    future['floor'] = 0
    forecast = m.predict(future)
    py.init_notebook_mode()

    fig = plot_plotly(m, forecast, xlabel='Date', ylabel='Confirmed Cases', uncertainty=True, figsize=(1100,600))  # This returns a plotly Figure
    fig.update_layout(title=province+', '+country+' Confirmed COVID-19 Cases and Forecast ('+str(t)+' day) as of' + str(NOW))
    
    c = cleanStr(country)
    p = cleanStr(province)
    os.makedirs("./images/" + c + "/" + NOW_FILE, exist_ok=True)
    fig.write_image("./images/" + c + "/" + NOW_FILE + "/" + NOW_FILE + "__" + c + "_" + p + "_forecast_" + str(t) + "_day.png")
    
    if disp:
        py.iplot(fig)


def generate_forecasts():
    for country in countries:
        print("Generating forecasts for: " + country)
        cases, totals = getDf(country)
        if totals.shape[0] < 3:
            continue
        with suppress_stdout_stderr():
            pred_country(country, 3, infectivity_factor=180, gMethod='linear', disp=False)
        sts = cases['Province/State'].unique()
        sts.sort()
        if len(sts) > 1:
            for st in sts:
                print("\t" + st)
                with suppress_stdout_stderr():
                    pred_province(country, st, 3, infectivity_factor=180, gMethod='linear', disp=False)

#generate_forecasts()

covid_19_ts = pd.read_csv(input_dir + 'covid_19_ts.csv')
covid_19_national_observations = pd.read_csv(input_dir + 'global/covid_19_national_observations.csv')
covid_19_infected_observations = pd.read_csv(input_dir + 'global/covid_19_infected_observations.csv')
covid_19_world_totals = pd.read_csv(input_dir + 'global/covid_19_world_totals.csv')

#correct date parsing on some of the JH data
covid_19_ts['Date'] = pd.to_datetime(covid_19_ts['Date'])
covid_19_national_observations['Date'] = pd.to_datetime(covid_19_national_observations['Date'])
covid_19_infected_observations['Date'] = pd.to_datetime(covid_19_infected_observations['Date'])
covid_19_world_totals['Date'] = pd.to_datetime(covid_19_world_totals['Date'])

world_chart = px.bar(covid_19_infected_observations, 
    x="Day", y="Active Cases", color="Country/Region", title="Global Active Cases by Day of Outbreak (Confirmed - Recovered - Deaths)")
world_chart.write_html('./www//global_by_day.html')

covid_19_world_totals_state = covid_19_world_totals[['Date', 'Active Cases', 'Recovered', 'Death']].melt(id_vars=['Date'], 
            value_vars=['Active Cases', 'Death', 'Recovered'], value_name="Population", var_name='Status')
world_chart = px.bar(covid_19_world_totals_state, 
    x="Date", y="Population", color="Status", title="Global Active, Recovered and Deaths by Date")
world_chart.write_html('./www/global.html')

fb_df = covid_19_world_totals[['Date', 'Active Cases']].copy()
fb_df = fb_df.sort_values('Date').reset_index(drop=True)
fb_df.columns = ['ds','y']
#print(fb_df)

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, growth='linear', n_changepoints=3)
m.fit(fb_df)

future = m.make_future_dataframe(periods=3)

forecast = m.predict(future)

fig = plot_plotly(m, forecast, xlabel='Date', ylabel='Active Cases', uncertainty=True)  # This returns a plotly Figure
fig.update_layout(title='Active Global COVID-19 Cases and Forecast')
fig.write_html('./www/global_forecast.html')

state_geo = os.path.join('./reference/', 'us-states.json')
state_data = pd.read_csv('./reference/COVID19-US-State-Quarantine.csv')
combined = pd.read_csv(input_dir + 'web_cases_state.csv')

state_data['Quarantined'] = 1
state_data.loc[state_data['US States Quarantined'] == 2, 'Quarantined'] = 0

combined['Last Update'] = pd.to_datetime(combined['Last_Update'])
combined['Latitude'] = combined['Lat']
combined['Longitude'] = combined['Long_']
combined['Province/State'] = combined['Province_State']
combined['Country/Region'] = combined['Country_Region']
combined = combined.sort_values('Last Update').reset_index(drop=True)

combined = combined[['Last Update', 'Latitude', 'Longitude', 'Country/Region', 'Province/State','Active', 'Confirmed','Recovered','Deaths']].copy().reset_index(drop=True)

latest = combined.sort_values('Last Update').groupby(['Latitude','Longitude']).tail(1).copy().reset_index(drop=True)
latest['Active'] = latest['Confirmed'] - latest['Recovered'] - latest['Deaths']

deaths = latest[latest['Deaths']>0]
deaths = pd.DataFrame(deaths.values.repeat(deaths.Deaths, axis=0), columns=deaths.columns)

confirmed = latest[latest['Confirmed']>0]
confirmed = pd.DataFrame(confirmed.values.repeat(confirmed.Confirmed, axis=0), columns=confirmed.columns)

us_confirmed = latest[latest['Confirmed']>0]
us_confirmed = us_confirmed[us_confirmed['Country/Region'] == 'US']
us_confirmed = pd.DataFrame(us_confirmed.values.repeat(us_confirmed.Confirmed, axis=0), columns=us_confirmed.columns)

recovered = latest[latest['Recovered']>0]
recovered = pd.DataFrame(recovered.values.repeat(recovered.Recovered, axis=0), columns=recovered.columns)

active = latest[latest['Active']>0]
active = pd.DataFrame(active.values.repeat(active.Active, axis=0), columns=active.columns)

us_active = latest[latest['Active']>0]
us_active = us_active[us_active['Country/Region'] == 'US']
us_active = pd.DataFrame(us_active.values.repeat(us_active.Active, axis=0), columns=us_active.columns)

na = latest[latest['Active'] > 0]
na_active = na[na['Country/Region'] == 'US']
na_active = na_active.append(na[na['Country/Region'] == 'Mexico'])
na_active = na_active.append(na[na['Country/Region'] == 'Canada'])
na_active = pd.DataFrame(na_active.values.repeat(na_active.Active, axis=0), columns=na_active.columns)

folium_map = folium.Map(location=[38.826555, -100.244867],
                        zoom_start=4,
                        tiles=None)
folium.TileLayer('CartoDB dark_matter', name='Base Layer').add_to(folium_map)
                             
#FastMarkerCluster(data=list(zip(deaths['Latitude'].values, deaths['Longitude'].values)), name='Reported Deaths').add_to(folium_map)
#FastMarkerCluster(data=list(zip(recovered['Latitude'].values, recovered['Longitude'].values)), name='Recovered Cases').add_to(folium_map)
#FastMarkerCluster(data=list(zip(active['Latitude'].values, active['Longitude'].values)), name='Active Cases').add_to(folium_map)
FastMarkerCluster(data=list(zip(na_active['Latitude'].values, na_active['Longitude'].values)), name='North America Active Cases').add_to(folium_map)

folium_map.choropleth(
    name='Quarantine in Effect',
    geo_data=state_geo,
    data=state_data,
    columns=['US States', 'Quarantined'],
    key_on='feature.properties.name',
    fill_color='RdPu',
    fill_opacity=0.7,
    line_opacity=0.2
)

lc = folium.LayerControl(collapsed=False).add_to(folium_map)

folium_map.save('./www/mapping.html')
