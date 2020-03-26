#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

import json

# for geocoding stuff
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from tqdm import tqdm

import glob
import os


# In[2]:


tqdm.pandas()


# In[3]:


output_dir = "./data/"
input_dir = "./csse_covid_19_data/csse_covid_19_daily_reports/"
extension = 'csv'
all_filenames = [i for i in glob.glob(input_dir+'*.{}'.format(extension))]

# %% combine em up
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(output_dir + "covid_19_raw.csv", index=False, encoding='utf-8-sig') 


# In[4]:


geopy.geocoders.options.default_timeout = 30
locator = Nominatim(user_agent="mesur.io")
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)


# In[5]:


# let's do a little data cleanup
combined_csv['Country/Region'] = combined_csv['Country/Region'].str.strip()
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Korea, South', 'South Korea')
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Republic of Korea', 'South Korea')
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Iran (Islamic Republic of)', 'Iran')
combined_csv['Country/Region'] = combined_csv['Country/Region'].replace('Mainland China', 'China')

combined_csv['Country_Region'] = combined_csv['Country_Region'].str.strip()
combined_csv['Country_Region'] = combined_csv['Country_Region'].replace('Korea, South', 'South Korea')
combined_csv['Country_Region'] = combined_csv['Country_Region'].replace('Republic of Korea', 'South Korea')
combined_csv['Country_Region'] = combined_csv['Country_Region'].replace('Iran (Islamic Republic of)', 'Iran')
combined_csv['Country_Region'] = combined_csv['Country_Region'].replace('Mainland China', 'China')


# In[6]:


combined = combined_csv.copy()


# In[7]:


locations = combined.groupby(['Country/Region', 'Province/State'])['Latitude', 'Longitude'].mean().reset_index()
locations.columns = ['Country/Region', 'Province/State', 'Latitude_Lookup', 'Longitude_Lookup']

combined = pd.merge(left=combined, right=locations, left_on=['Country/Region', 'Province/State'], right_on=['Country/Region', 'Province/State'], how='left')
combined['Latitude'] = combined['Latitude'].fillna(combined['Latitude_Lookup'])
combined['Longitude'] = combined['Longitude'].fillna(combined['Longitude_Lookup'])
del combined['Latitude_Lookup'] 
del combined['Longitude_Lookup'] 

locations2 = combined.groupby(['Country/Region', 'Province/State'])['Lat', 'Long_'].mean().reset_index()
locations2.columns = ['Country/Region', 'Province/State', 'Latitude_Lookup', 'Longitude_Lookup']

combined = pd.merge(left=combined, right=locations2, left_on=['Country/Region', 'Province/State'], right_on=['Country/Region', 'Province/State'], how='left')
combined['Latitude'] = combined['Latitude'].fillna(combined['Latitude_Lookup'])
combined['Longitude'] = combined['Longitude'].fillna(combined['Longitude_Lookup'])
del combined['Latitude_Lookup'] 
del combined['Longitude_Lookup'] 

combined['Last Update'] = combined['Last Update'].fillna(combined['Last_Update'])
combined['Country/Region'] = combined['Country/Region'].fillna(combined['Country_Region'])
combined['Province/State'] = combined['Province/State'].fillna(combined['Province_State'])
combined['Province/State'] = combined['Province/State'].fillna(combined['Country/Region'])
combined['Confirmed'] = combined['Confirmed'].fillna(0)
combined['Deaths'] = combined['Deaths'].fillna(0)


# In[8]:


combined['Last Update'] = pd.to_datetime(combined['Last Update'])
combined = combined.sort_values('Last Update').reset_index(drop=True)


# In[9]:


combined['Geo_Input'] = combined['Province/State']+', '+combined['Country/Region'] 


# In[10]:


non_located = combined[combined['Latitude'].isna()]
non_located = non_located[non_located['Province/State'] != 'Cruise Ship']


# In[11]:


geo_inputs = non_located['Geo_Input'].unique()
combined['Location_Key_Raw'] = combined.apply(lambda x: (x.Latitude, x.Longitude), axis = 1)
#for testing you may want to trim this down a bit
#geo_inputs = geo_inputs[:10]


# In[12]:


def geocode():
    print('Geocoding for: ', len(geo_inputs), 'locations')
    #use progress_apply() for interactive progress
    d = dict(zip(geo_inputs, pd.Series(geo_inputs).apply(geocode).apply(lambda x: (x.latitude if pd.notnull(x.latitude) else x.latitude, 
                                                                                   x.longitude if pd.notnull(x.longitude) else x.longitude) if pd.notnull(x) else x)
                )
            )
    pd.DataFrame.from_dict(d, orient="index").to_csv('./reference/geoloc_dict.json')


# In[13]:


d = pd.read_csv('./reference/geoloc_dict.json', index_col=0).to_dict("split")
d = dict(zip(d["index"], d["data"]))


# In[14]:


combined['Location_Key'] = combined['Geo_Input'].map(d)


# In[15]:


combined['Location_Key'] = combined['Location_Key'].fillna(combined['Location_Key_Raw'])


# In[16]:


combined['Latitude'] = combined.loc[combined['Latitude'].isna(), 'Location_Key'].apply(lambda x: x[0])
combined['Longitude'] = combined.loc[combined['Longitude'].isna(), 'Location_Key'].apply(lambda x: x[1])


# In[17]:


#let's do a recovery est
# first need the day of outbreak
combined = combined.sort_values('Last Update').reset_index(drop=True)
combined['Day'] = combined.groupby('Country/Region').cumcount()
combined['DayLoc'] = combined.groupby(['Latitude','Longitude']).cumcount()
combined['DayCountry'] = combined.groupby('Country/Region').cumcount()
combined['DayCountryProvince'] = combined.groupby('Geo_Input').cumcount()

combined['UnknownActive'] = combined['Confirmed'] - combined['Deaths']
combined['RecoveredEst'] = np.floor(combined['UnknownActive'] * .14)
combined['Recovered'] = combined['Recovered'].fillna(0)
#hold on this for now, there is a formula that is curve based for this
#combined.loc[combined['Recovered'] == 0, 'Recovered'] = combined['RecoveredEst']
combined['Active'] = combined['UnknownActive'] - combined['Recovered']


# In[18]:


combined = combined.fillna(0)


# In[19]:


# do a little reording and subselection
combined_csv = combined[['Last Update','Latitude','Longitude','Country/Region','Province/State','FIPS','Admin2',
                         'Confirmed','Deaths','Recovered','UnknownActive', 'Active',
                         'Day','DayLoc','DayCountry','DayCountryProvince']]
combined_csv = combined_csv.sort_values(['Last Update','Latitude','Longitude','Country/Region','Province/State'])
#combined_csv


# In[20]:


combined_csv.to_csv(output_dir + "combined.csv", index=False, encoding='utf-8-sig')


# In[21]:


web_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases.csv')
web_cases_state = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_state.csv')
web_cases_country = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
web_cases_time = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv')

web_cases.to_csv(output_dir + "web_cases.csv", index=False, encoding='utf-8-sig')
web_cases_state.to_csv(output_dir + "web_cases_state.csv", index=False, encoding='utf-8-sig')
web_cases_country.to_csv(output_dir + "web_cases_country.csv", index=False, encoding='utf-8-sig')
web_cases_time.to_csv(output_dir + "web_cases_time.csv", index=False, encoding='utf-8-sig')


# In[ ]:





# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


#john's hopkins raw files
ts_deaths = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
ts_confirmed = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
#new recovered tracking has now been dropped :(
ts_recovered = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

# let's unpivot that nasty excel style stuff
ts_deaths = pd.melt(ts_deaths, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Observation')
ts_deaths['Observation Type'] = 'Death'
ts_confirmed = pd.melt(ts_confirmed, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Observation')
ts_confirmed['Observation Type'] = 'Confirmed'
ts_recovered = pd.melt(ts_recovered, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Observation')
ts_recovered['Observation Type'] = 'Recovered'

ts_deaths['Date'] = pd.to_datetime(ts_deaths['Date'])
ts_confirmed['Date'] = pd.to_datetime(ts_confirmed['Date'])
ts_recovered['Date'] = pd.to_datetime(ts_recovered['Date'])

#and concat into one nice set
covid_19_ts = ts_deaths.copy()
covid_19_ts = covid_19_ts.append(ts_recovered)
covid_19_ts = covid_19_ts.append(ts_confirmed)
covid_19_ts = covid_19_ts.sort_values(['Country/Region', 'Province/State', 'Date']).reset_index(drop=True)

#now drop 0 values
covid_19_ts = covid_19_ts[covid_19_ts['Observation'] != 0]


# In[27]:


overallDf.to_csv(output_dir + "covid_19_by_date_and_country.csv", index=False, encoding='utf-8-sig')
covid_19_ts.to_csv(output_dir + "covid_19_ts.csv", index=False, encoding='utf-8-sig')


# In[28]:


#display for debug
#display(covid_19_ts)


# In[ ]:




