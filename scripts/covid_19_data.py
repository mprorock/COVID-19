#!/usr/bin/env python
# coding: utf-8

# In[78]:


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


# In[79]:


tqdm.pandas()


# In[80]:


output_dir = "./data/"
input_dir = "./csse_covid_19_data/csse_covid_19_daily_reports/"
extension = 'csv'
all_filenames = [i for i in glob.glob(input_dir+'*.{}'.format(extension))]

# %% combine em up
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(output_dir + "covid_19_raw.csv", index=False, encoding='utf-8-sig') 


# In[81]:


geopy.geocoders.options.default_timeout = 30
locator = Nominatim(user_agent="mesur.io")
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)


# In[82]:


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


# In[83]:


combined = combined_csv.copy()


# In[84]:


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


# In[85]:


combined['Last Update'] = pd.to_datetime(combined['Last Update'])
combined = combined.sort_values('Last Update').reset_index(drop=True)


# In[86]:


combined['Geo_Input'] = combined['Province/State']+', '+combined['Country/Region'] 


# In[87]:


non_located = combined[combined['Latitude'].isna()]
non_located = non_located[non_located['Province/State'] != 'Cruise Ship']


# In[88]:


geo_inputs = non_located['Geo_Input'].unique()
combined['Location_Key_Raw'] = combined.apply(lambda x: (x.Latitude, x.Longitude), axis = 1)
#for testing you may want to trim this down a bit
#geo_inputs = geo_inputs[:10]


# In[89]:


def geocode():
    print('Geocoding for: ', len(geo_inputs), 'locations')
    #use progress_apply() for interactive progress
    d = dict(zip(geo_inputs, pd.Series(geo_inputs).apply(geocode).apply(lambda x: (x.latitude if pd.notnull(x.latitude) else x.latitude, 
                                                                                   x.longitude if pd.notnull(x.longitude) else x.longitude) if pd.notnull(x) else x)
                )
            )
    pd.DataFrame.from_dict(d, orient="index").to_csv('./reference/geoloc_dict.json')


# In[90]:


d = pd.read_csv('./reference/geoloc_dict.json', index_col=0).to_dict("split")
d = dict(zip(d["index"], d["data"]))


# In[91]:


combined['Location_Key'] = combined['Geo_Input'].map(d)


# In[92]:


combined['Location_Key'] = combined['Location_Key'].fillna(combined['Location_Key_Raw'])


# In[93]:


combined['Latitude'] = combined.loc[combined['Latitude'].isna(), 'Location_Key'].apply(lambda x: x[0])
combined['Longitude'] = combined.loc[combined['Longitude'].isna(), 'Location_Key'].apply(lambda x: x[1])


# In[99]:


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


# In[100]:


combined = combined.fillna(0)


# In[102]:


# do a little reording and subselection
combined_csv = combined[['Last Update','Latitude','Longitude','Country/Region','Province/State','FIPS','Admin2',
                         'Confirmed','Deaths','Recovered','UnknownActive', 'Active',
                         'Day','DayLoc','DayCountry','DayCountryProvince']]
combined_csv = combined_csv.sort_values(['Last Update','Latitude','Longitude','Country/Region','Province/State'])
#combined_csv


# In[103]:


combined_csv.to_csv(output_dir + "combined.csv", index=False, encoding='utf-8-sig')


# In[104]:


web_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases.csv')
web_cases_state = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_state.csv')
web_cases_country = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
web_cases_time = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv')

web_cases.to_csv(output_dir + "web_cases.csv", index=False, encoding='utf-8-sig')
web_cases_state.to_csv(output_dir + "web_cases_state.csv", index=False, encoding='utf-8-sig')
web_cases_country.to_csv(output_dir + "web_cases_country.csv", index=False, encoding='utf-8-sig')
web_cases_time.to_csv(output_dir + "web_cases_time.csv", index=False, encoding='utf-8-sig')


# In[105]:


us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'D.C.',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Northern Mariana Islands':'MP',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Palau': 'PW',
        'Pennsylvania': 'PA',
        'Puerto Rico': 'PR',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virgin Islands': 'VI',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY',
    }


# In[ ]:





# In[106]:


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


# In[107]:


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


# In[108]:


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


# In[109]:


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


# In[110]:


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


# In[111]:


overallDf.to_csv(output_dir + "covid_19_by_date_and_country.csv", index=False, encoding='utf-8-sig')
covid_19_ts.to_csv(output_dir + "covid_19_ts.csv", index=False, encoding='utf-8-sig')


# In[112]:


#display for debug
#display(covid_19_ts)


# In[113]:


# now run any JH combinations


# In[114]:


covid_19_by_country_and_day_of_outbreak = pd.read_csv(output_dir + 'covid_19_by_date_and_country.csv')
covid_19_by_date = pd.read_csv(output_dir + 'covid_19_by_date.csv')
covid_19_by_day = pd.read_csv(output_dir + 'covid_19_by_day.csv')
covid_19_overall = pd.read_csv(output_dir + 'covid_19.csv')
covid_19_ts = pd.read_csv(output_dir + 'covid_19_ts.csv')


# In[115]:


covid_19_ts['Date'] = pd.to_datetime(covid_19_ts['Date'])


# In[116]:


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

covid_19_infected_observations['Case Bins'] = pd.qcut(covid_19_infected_observations['Active Cases'], 10)
maxes = covid_19_infected_observations.groupby('Country/Region')['Day'].max().reset_index()
maxes.columns = ['Country/Region', 'Max Day']
covid_19_infected_observations = pd.merge(left=covid_19_infected_observations, right=maxes, left_on='Country/Region', right_on='Country/Region', how='left')


# In[117]:


covid_19_national_observations.to_csv(output_dir + "global/covid_19_national_observations.csv", index=False, encoding="utf-8-sig")
covid_19_infected_observations.to_csv(output_dir + "global/covid_19_infected_observations.csv", index=False, encoding="utf-8-sig")


# In[118]:


#covid_19_national_observations[covid_19_national_observations['Country/Region']=='Brazil']


# In[119]:


world_totals = covid_19_national_observations.groupby(['Date']).sum().reset_index().copy()
world_totals = world_totals.sort_values('Date')
world_totals['Total Confirmed'] = world_totals['Confirmed'].rolling(1).sum().fillna(0)
world_totals['Total Deaths'] = world_totals['Death'].rolling(1).sum().fillna(0)
world_totals['Total Recovered'] = world_totals['Recovered'].rolling(1).sum().fillna(0)
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


# In[120]:


#country specific stuff below here


# In[128]:


def cleanStr(s):
    c = s.lower().replace(' ', '_')
    c = c.replace('*', '').replace('(', '_').replace(')', '_').replace(',', '_')
    c = c.strip()
    return c

def partitionByCountry(country):  
    print('Processing data files for', country)
    c = cleanStr(country)
    os.makedirs(output_dir + 'countries/'+c, exist_ok=True)
    
    df = combined_csv[combined_csv['Country/Region'] == country].copy()
    #add some backwards compat
    df['Date'] = df['Last Update']
    df['Death'] = df['Deaths']

    df['Province/State'] = df['Province/State'].str.rsplit(',').str[-1].str.strip() 
    df['Province/State'] = df['Province/State'].replace(us_state_abbrev)

    df = df.sort_values(['Province/State', 'Date'])
    
    if df['Province/State'].count() > 1:
        print(df['Province/State'].unique())

    cases = pd.DataFrame()
    if df['Province/State'].count() == 0:
        df['Province/State'] = country
    cases = df.groupby(['Date', 'Province/State']).sum().reset_index()
    
    totals = cases.groupby(['Date']).sum().reset_index().copy()
    totals = totals.sort_values('Date')    
    
    totals['Total Confirmed'] = totals['Confirmed'].rolling(1).sum().fillna(0)
    totals['Total Deaths'] = totals['Death'].rolling(1).sum().fillna(0)
    totals['Total Recovered'] = totals['Recovered'].rolling(1).sum().fillna(0)
    totals['Active Cases'] = totals['Total Confirmed'] - totals['Total Deaths'] - totals['Total Recovered']
    totals['Death Rate'] = np.round(totals['Total Deaths'] / totals['Total Confirmed'], 3).fillna(0)
    totals['Death Change Rate'] = np.round(totals['Total Deaths'].pct_change(), 3).fillna(0)
    totals['Recovery Change Rate'] = np.round(totals['Total Recovered'].pct_change(), 3).fillna(0)
    totals['Confirmed Change Rate'] = np.round(totals['Total Confirmed'].pct_change(), 3).fillna(0)
    totals['Confirmed Rolling 3 Change Rate'] = totals['Confirmed Change Rate'].rolling(3).mean().fillna(0)
    
    totals['New Active Cases'] = totals['Active Cases'] - totals['Active Cases'].shift()
    totals['New Active Cases PCT Change'] = totals['New Active Cases'].pct_change().fillna(0)

    totals['New Cases'] = totals['Confirmed'] - totals['Confirmed'].shift()
    totals['New Case PCT Change'] = totals['New Cases'].pct_change().fillna(0)

    totals['New Deaths'] = totals['Death'] - totals['Death'].shift()
    totals['New Death PCT Change'] = totals['New Deaths'].pct_change().fillna(0)
    
    totals['New Recovered'] = totals['Recovered'] - totals['Recovered'].shift()
    totals['New Recovered PCT Change'] = totals['New Recovered'].pct_change().fillna(0)
    
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

        
    cases.to_csv(output_dir + 'countries/'+c+'/covid_19_'+c+'_cases.csv', index=False, encoding='utf-8-sig')
    totals.to_csv(output_dir + 'countries/'+c+'/covid_19_'+c+'_totals.csv', index=False, encoding='utf-8-sig')


# In[129]:


countries = combined_csv['Country/Region'].unique()

countries.sort()
for cnt in countries:
    partitionByCountry(cnt)


# In[ ]:




