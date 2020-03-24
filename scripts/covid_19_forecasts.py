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
    if cases['Confirmed'].sum() < 5:
        return False
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

generate_forecasts()
