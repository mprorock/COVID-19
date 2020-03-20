#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

import glob
import os


# In[2]:


output_dir = "./data/"
input_dir = "./csse_covid_19_data/csse_covid_19_daily_reports/"
extension = 'csv'
all_filenames = [i for i in glob.glob(input_dir+'*.{}'.format(extension))]

# %% combine em up
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(output_dir + "covid_19.csv", index=False, encoding='utf-8-sig') 

# let's do a little data cleanup
combined_csv['Country/Region'] = combined_csv['Country/Region'].str.strip()
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Korea, South', 'South Korea')
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Republic of Korea', 'South Korea')
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Iran (Islamic Republic of)', 'Iran')
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Mainland China', 'China')


# In[3]:


df = combined_csv.copy()
df['Last Update'] = pd.to_datetime(df['Last Update']).dt.date 

def firsti(Series, offset):
    return Series.first(offset)

df = df.groupby(by=['Last Update', 'Country/Region'])[
    #'Last Update', 'Country/Region',
    'Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df = df.sort_values('Last Update', ascending=True).reset_index()
df['Active Cases'] = df['Confirmed'] - df['Recovered'] - df['Deaths']
df['Cases'] = df['Confirmed'] - df['Recovered'] 
df['Death Rate'] = df['Deaths'] / df['Confirmed']
df['Recovery Rate'] = df['Recovered'] / df['Confirmed']
df['New Deaths'] = df['Deaths'] - df['Deaths'].shift()
df['New Recovered'] = df['Recovered'] - df['Recovered'].shift()
df['New Cases'] = df['Confirmed'] - df['Confirmed'].shift()
df['New Case Rate'] = df['New Cases'].pct_change()
df['New Death Rate'] = df['New Deaths'].pct_change()
df['Last Update'] = pd.to_datetime(df['Last Update'])
df['Date'] = pd.DatetimeIndex(df['Last Update']).astype ( np.int64 )/1000000
df['Day'] = df.groupby('Country/Region').cumcount()
df = df.dropna().reset_index()

df.to_csv(output_dir + "covid_19_by_date_and_country.csv", index=False, encoding='utf-8-sig')


# In[4]:


overallDf = df.copy().groupby('Last Update').agg({
    'Confirmed':'sum',
    'Deaths':'sum',
    'Recovered':'sum'
    }).reset_index()
overallDf = overallDf.sort_values('Last Update', ascending=True)
overallDf['Active Cases'] = overallDf['Confirmed'] - overallDf['Recovered'] - overallDf['Deaths']
overallDf['Cases'] = overallDf['Confirmed'] - overallDf['Recovered'] 
overallDf['Death Rate'] = overallDf['Deaths'] / overallDf['Confirmed']
overallDf['Recovery Rate'] = overallDf['Recovered'] / overallDf['Confirmed']
overallDf['New Deaths'] = overallDf['Deaths'] - overallDf['Deaths'].shift()
overallDf['New Recovered'] = overallDf['Recovered'] - overallDf['Recovered'].shift()
overallDf['New Cases'] = overallDf['Confirmed'] - overallDf['Confirmed'].shift()
overallDf['New Case Rate'] = overallDf['New Cases'].pct_change()
overallDf['New Death Rate'] = overallDf['New Deaths'].pct_change()
overallDf['Date'] = pd.DatetimeIndex(overallDf['Last Update']).astype ( np.int64 )/1000000
overallDf = overallDf.dropna().reset_index()

overallDf.to_csv(output_dir + "covid_19_by_date.csv", index=False, encoding='utf-8-sig')


# In[5]:


overallDf = df.copy().groupby('Day').agg({
    'Confirmed':'sum',
    'Deaths':'sum',
    'Recovered':'sum'
    }).reset_index()
overallDf = overallDf.sort_values('Day', ascending=True)
overallDf['Active Cases'] = overallDf['Confirmed'] - overallDf['Recovered'] - overallDf['Deaths']
overallDf['Cases'] = overallDf['Confirmed'] - overallDf['Recovered'] 
overallDf['Death Rate'] = overallDf['Deaths'] / overallDf['Confirmed']
overallDf['Recovery Rate'] = overallDf['Recovered'] / overallDf['Confirmed']
overallDf['New Deaths'] = overallDf['Deaths'] - overallDf['Deaths'].shift()
overallDf['New Recovered'] = overallDf['Recovered'] - overallDf['Recovered'].shift()
overallDf['New Cases'] = overallDf['Confirmed'] - overallDf['Confirmed'].shift()
overallDf['New Case Rate'] = overallDf['New Cases'].pct_change()
overallDf['New Death Rate'] = overallDf['New Deaths'].pct_change()
overallDf = overallDf.dropna().reset_index()

overallDf.to_csv(output_dir + "covid_19_by_day.csv", index=False, encoding='utf-8-sig')


# In[6]:


overallDf = df.copy().groupby(by=['Country/Region','Day']).agg({
    'Confirmed':'sum',
    'Deaths':'sum',
    'Recovered':'sum'
    }).reset_index()
overallDf = overallDf.sort_values('Day', ascending=True)
overallDf['Active Cases'] = overallDf['Confirmed'] - overallDf['Recovered'] - overallDf['Deaths']
overallDf['Cases'] = overallDf['Confirmed'] - overallDf['Recovered'] 
overallDf['Death Rate'] = overallDf['Deaths'] / overallDf['Confirmed']
overallDf['Recovery Rate'] = overallDf['Recovered'] / overallDf['Confirmed']
overallDf['New Deaths'] = overallDf['Deaths'] - overallDf['Deaths'].shift()
overallDf['New Recovered'] = overallDf['Recovered'] - overallDf['Recovered'].shift()
overallDf['New Cases'] = overallDf['Confirmed'] - overallDf['Confirmed'].shift()
overallDf['New Case Rate'] = overallDf['New Cases'].pct_change()
overallDf['New Death Rate'] = overallDf['New Deaths'].pct_change()
overallDf = overallDf.dropna().reset_index()


# In[7]:


#john's hopkins raw files
ts_deaths = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
ts_recovered = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
ts_confirmed = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

# let's unpivot that nasty excel style stuff
ts_deaths = pd.melt(ts_deaths, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Observation')
ts_deaths['Observation Type'] = 'Death'
ts_recovered = pd.melt(ts_recovered, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Observation')
ts_recovered['Observation Type'] = 'Recovered'
ts_confirmed = pd.melt(ts_confirmed, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Observation')
ts_confirmed['Observation Type'] = 'Confirmed'

ts_deaths['Date'] = pd.to_datetime(ts_deaths['Date'])
ts_recovered['Date'] = pd.to_datetime(ts_recovered['Date'])
ts_confirmed['Date'] = pd.to_datetime(ts_confirmed['Date'])

#and concat into one nice set
covid_19_ts = ts_deaths.copy()
covid_19_ts = covid_19_ts.append(ts_recovered)
covid_19_ts = covid_19_ts.append(ts_confirmed)
covid_19_ts = covid_19_ts.sort_values(['Country/Region', 'Date']).reset_index(drop=True)
#now drop 0 values
covid_19_ts = covid_19_ts[covid_19_ts['Observation'] != 0]


# In[8]:


overallDf.to_csv(output_dir + "covid_19_by_date_and_country.csv", index=False, encoding='utf-8-sig')
covid_19_ts.to_csv(output_dir + "covid_19_ts.csv", index=False, encoding='utf-8-sig')


# In[9]:


#display for debug
# display(covid_19_ts)


# In[10]:


# now run any JH combinations


# In[11]:


covid_19_by_country_and_day_of_outbreak = pd.read_csv(output_dir + 'covid_19_by_date_and_country.csv')
covid_19_by_date = pd.read_csv(output_dir + 'covid_19_by_date.csv')
covid_19_by_day = pd.read_csv(output_dir + 'covid_19_by_day.csv')
covid_19_overall = pd.read_csv(output_dir + 'covid_19.csv')
covid_19_ts = pd.read_csv(output_dir + 'covid_19_ts.csv')


# In[12]:


covid_19_ts['Date'] = pd.to_datetime(covid_19_ts['Date'])


# In[13]:


ts_df = covid_19_ts.copy()
#if you just want to include some, exclude others, see line below:
#ts_df = covid_19_ts[covid_19_ts['Country/Region'] != 'US'].copy()
ts_df = ts_df.drop('Province/State', axis=1)
ts_df = ts_df.sort_values(['Country/Region', 'Date'])

covid_19_national_observations = ts_df.groupby(['Date', 'Country/Region', 'Observation Type'])['Observation'].sum().reset_index()
covid_19_national_observations = covid_19_national_observations.pivot_table(columns='Observation Type', index=['Date', 'Country/Region'], values='Observation').reset_index().rename_axis(None, axis=1).fillna(0)

#let's get a copy for post infection as well
covid_19_infected_observations = covid_19_national_observations.copy()
covid_19_infected_observations = covid_19_infected_observations[covid_19_infected_observations['Confirmed'] >= 20].fillna(0)

#on to key values for reporting
covid_19_national_observations['Day'] = covid_19_national_observations.groupby('Country/Region')['Confirmed'].cumcount()
covid_19_national_observations['Day'] = covid_19_national_observations.groupby('Country/Region')['Confirmed'].fillna(method='bfill')

covid_19_national_observations['Active Cases'] = covid_19_national_observations['Confirmed'] - covid_19_national_observations['Recovered']  - covid_19_national_observations['Death'] 

covid_19_national_observations['Likely Cases 1pct'] = covid_19_national_observations['Death'] * 100
covid_19_national_observations['Likely Cases 1.8pct'] = covid_19_national_observations['Death'] * 180
covid_19_national_observations['Likely Cases 3.5pct'] = covid_19_national_observations['Death'] * 350

covid_19_national_observations['Death Rate'] = np.round(covid_19_national_observations['Death'] / covid_19_national_observations['Confirmed'],3)
covid_19_national_observations['Death Change Rate'] = covid_19_national_observations.groupby('Country/Region')['Death'].pct_change()
covid_19_national_observations['Recovery Change Rate'] = covid_19_national_observations.groupby('Country/Region')['Recovered'].pct_change()
covid_19_national_observations['Confirmed Change Rate'] = covid_19_national_observations.groupby('Country/Region')['Confirmed'].pct_change()

#now for same values on limited set
covid_19_infected_observations['Day'] = covid_19_infected_observations.groupby('Country/Region', squeeze=True).cumcount()

covid_19_infected_observations['Active Cases'] = covid_19_infected_observations['Confirmed'] - covid_19_infected_observations['Recovered']  - covid_19_infected_observations['Death'] 

covid_19_infected_observations['Likely Cases 1pct'] = covid_19_infected_observations['Death'] * 100
covid_19_infected_observations['Likely Cases 1.8pct'] = covid_19_infected_observations['Death'] * 180
covid_19_infected_observations['Likely Cases 3.5pct'] = covid_19_infected_observations['Death'] * 350

covid_19_infected_observations['Death Rate'] = np.round(covid_19_infected_observations['Death'] / covid_19_infected_observations['Confirmed'],3)
covid_19_infected_observations['Death Change Rate'] = covid_19_infected_observations.groupby('Country/Region')['Death'].pct_change()
covid_19_infected_observations['Recovery Change Rate'] = covid_19_infected_observations.groupby('Country/Region')['Recovered'].pct_change()
covid_19_infected_observations['Confirmed Change Rate'] = covid_19_infected_observations.groupby('Country/Region')['Confirmed'].pct_change()


# In[14]:


covid_19_national_observations.to_csv(output_dir + "global/covid_19_national_observations.csv", index=False, encoding="utf-8-sig")
covid_19_infected_observations.to_csv(output_dir + "global/covid_19_infected_observations.csv", index=False, encoding="utf-8-sig")


# In[15]:


covid_19_national_observations[covid_19_national_observations['Country/Region']=='Brazil']


# In[16]:


world_totals = covid_19_national_observations.groupby(['Date']).sum().reset_index().copy()
world_totals = world_totals.sort_values('Date')
world_totals['Total Confirmed'] = world_totals['Confirmed'].rolling(1).sum()
world_totals['Total Deaths'] = world_totals['Death'].rolling(1).sum()
world_totals['Total Recovered'] = world_totals['Recovered'].rolling(1).sum()
world_totals['Active Cases'] = world_totals['Total Confirmed'] - world_totals['Total Deaths'] - world_totals['Total Recovered']
world_totals['Death Rate'] = np.round(world_totals['Total Deaths'] / world_totals['Total Confirmed'],3)
world_totals['Death Change Rate'] = world_totals['Total Deaths'].pct_change()
world_totals['Recovery Change Rate'] = world_totals['Total Recovered'].pct_change()
world_totals['Confirmed Change Rate'] = world_totals['Total Confirmed'].pct_change()

world_totals['Likely Cases C86'] = world_totals['Active Cases'] * 1.14
world_totals['Likely Cases 1pct'] = world_totals['Total Deaths'] * 100
world_totals['Likely Cases 1.8pct'] = world_totals['Total Deaths'] * 180
world_totals['Likely Cases 3.5pct'] = world_totals['Total Deaths'] * 350

world_totals.to_csv(output_dir + "global/covid_19_world_totals.csv", index=False, encoding='utf-8-sig')


# In[17]:


#country specific stuff below here


# In[18]:


def partitionByCountry(country):  
    print('Processing data files for', country)
    c = country.lower().replace(' ', '_')
    c = c.replace('*', '').replace('(', '_').replace(')', '_')
    os.makedirs(output_dir + 'countries/'+c, exist_ok=True)

    df = covid_19_ts[covid_19_ts['Country/Region'] == country].copy()
    df = df.sort_values(['Province/State', 'Date'])

    cases = pd.DataFrame()
    if df['Province/State'].count() == 0:
        df['Province/State'] = country
        cases = df.groupby(['Date', 'Observation Type'])['Observation'].sum().reset_index()
        cases = cases.pivot_table(columns='Observation Type', index=['Date'], values='Observation').reset_index().rename_axis(None, axis=1)
        cases['Province/State'] = country
    else:
        cases = df.groupby(['Date', 'Province/State', 'Observation Type'])['Observation'].sum().reset_index()
        cases = cases.pivot_table(columns='Observation Type', index=['Date', 'Province/State'], values='Observation').reset_index().rename_axis(None, axis=1)
    
    # check if columns exist
    if 'Death' not in cases.columns:
        cases['Death'] = 0
    if 'Confirmed' not in cases.columns:
        cases['Death'] = 0
    if 'Recovered' not in cases.columns:
        cases['Recovered'] = 0
    
    totals = cases.groupby(['Date']).sum().reset_index().copy()
    totals = totals.sort_values('Date')    
    
    totals['Total Confirmed'] = totals['Confirmed'].rolling(1).sum()
    totals['Total Deaths'] = totals['Death'].rolling(1).sum()
    totals['Total Recovered'] = totals['Recovered'].rolling(1).sum()
    totals['Active Cases'] = totals['Total Confirmed'] - totals['Total Deaths'] - totals['Total Recovered']
    totals['Death Rate'] = np.round(totals['Total Deaths'] / totals['Total Confirmed'], 3)
    totals['Death Change Rate'] = np.round(totals['Total Deaths'].pct_change(), 3)
    totals['Recovery Change Rate'] = np.round(totals['Total Recovered'].pct_change(), 3)
    totals['Confirmed Change Rate'] = np.round(totals['Total Confirmed'].pct_change(), 3)
    
    totals['New Active Cases'] = totals['Active Cases'] - totals['Active Cases'].shift()
    totals['New Active Cases PCT Change'] = totals['New Active Cases'].pct_change()

    totals['New Cases'] = totals['Confirmed'] - totals['Confirmed'].shift()
    totals['New Case PCT Change'] = totals['New Cases'].pct_change()

    totals['New Deaths'] = totals['Death'] - totals['Death'].shift()
    totals['New Death PCT Change'] = totals['New Deaths'].pct_change()
    
    totals['New Recovered'] = totals['Recovered'] - totals['Recovered'].shift()
    totals['New Recovered PCT Change'] = totals['New Recovered'].pct_change()
    
    totals['Likely Cases C86'] = totals['Active Cases'] * 1.14
    totals['Likely Cases 1pct'] = totals['Total Deaths'] * 100
    totals['Likely Cases 1.8pct'] = totals['Total Deaths'] * 180
    totals['Likely Cases 3.5pct'] = totals['Total Deaths'] * 350
    
    if df['Province/State'].count() <= 1:
        cases['Total Confirmed'] = cases['Confirmed'].rolling(1).sum()
        cases['Total Deaths'] = cases['Death'].rolling(1).sum()
        cases['Total Recovered'] = cases['Recovered'].rolling(1).sum()
    else:
        cases['Total Confirmed'] = cases.groupby('Province/State')['Confirmed'].rolling(1).sum().reset_index(0,drop=True)
        cases['Total Deaths'] = cases.groupby('Province/State')['Death'].rolling(1).sum().reset_index(0,drop=True)
        cases['Total Recovered'] = cases.groupby('Province/State')['Recovered'].rolling(1).sum().reset_index(0,drop=True)
    cases['Active Cases'] = totals['Total Confirmed'] - totals['Total Deaths'] - totals['Total Recovered']
    cases['Death Rate'] = np.round(totals['Total Deaths'] / totals['Total Confirmed'],3)
    cases['Death Change Rate'] = np.round(totals['Total Deaths'].pct_change(), 3)
    cases['Recovery Change Rate'] = np.round(totals['Total Recovered'].pct_change(), 3)
    cases['Confirmed Change Rate'] = np.round(totals['Total Confirmed'].pct_change(), 3)

    
    cases['Likely Cases 1pct'] = np.round(cases['Death'] * 100)
    cases['Likely Cases 1.8pct'] = np.round(cases['Death'] * 180)
    cases['Likely Cases 3.5pct'] = np.round(cases['Death'] * 350)
    cases['Likely Cases C86'] = np.round(cases['Confirmed'] * 1.14)

    
    if c == 'us':
        states_abbrv=pd.read_csv('./reference/US-state-abbrv.csv')
        states=pd.DataFrame()
        states['Province/State'] = cases['Province/State'].drop_duplicates().str.rsplit(',').str[-1].str.strip()
        states_abbrv['Abbreviation'] = states_abbrv['Abbreviation'].str.strip()
        states_abbrv['State'] = states_abbrv['State'].str.strip()
        states = pd.merge(left=states, right=states_abbrv, left_on='Province/State', right_on='Abbreviation', how='left')
        states['State'] = states['State'].fillna(states['Province/State'])
        
        states.to_csv(output_dir + "countries/us/state_list.csv", index=False)
    
    cases.to_csv(output_dir + 'countries/'+c+'/covid_19_'+c+'_cases.csv', index=False, encoding='utf-8-sig')
    totals.to_csv(output_dir + 'countries/'+c+'/covid_19_'+c+'_totals.csv', index=False, encoding='utf-8-sig')


# In[19]:


countries = covid_19_national_observations['Country/Region'].unique()
for cnt in countries:
    partitionByCountry(cnt)


# In[ ]:




