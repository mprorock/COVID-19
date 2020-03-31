#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import pickle

from tqdm import tqdm

import glob
import os


# In[ ]:





# In[ ]:


output_dir = "./data/"
input_dir = "./csse_covid_19_data/csse_covid_19_daily_reports/"
extension = 'csv'
all_filenames = [i for i in glob.glob(input_dir+'*.{}'.format(extension))]

# %% combine em up
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(output_dir + "covid_19_raw.csv", index=False, encoding='utf-8-sig') 


# In[ ]:





# In[ ]:


def calc_timeseries_by_group(df, group_col='Location'):
    tdf = df.copy()
    tdf=tdf.sort_values('Last Update')
    timeseries_by_location = tdf.groupby(group_col)
    
    for days_shift in [1,3,7]:
        for orig_column in ['Recovered', 'Deaths', 'Confirmed', 'Active']:
            tdf[f'{days_shift}d new {orig_column}'] = timeseries_by_location[orig_column].diff(periods=days_shift)

    day_location_reached_100 = tdf[tdf['Confirmed']>100].groupby(group_col)['Last Update'].min().to_dict()
    day_location_reached_100_active = tdf[tdf['Active']>100].groupby(group_col)['Last Update'].min().to_dict()
    
    day_location_reached_1_deaths = tdf[tdf['Deaths']>1].groupby(group_col)['Last Update'].min().to_dict()
    day_location_reached_10_deaths = tdf[tdf['Deaths']>10].groupby(group_col)['Last Update'].min().to_dict()
    day_location_reached_100_deaths = tdf[tdf['Deaths']>100].groupby(group_col)['Last Update'].min().to_dict()
    #day_location_reached_1_per_100k = tdf[tdf['Confirmed per 100k capita']>1].groupby(group_col)['Last Update'].min().to_dict()

    def shift_dates(row, offset_by_location):
        date = row['Last Update']
        location = row[group_col]
        if location in offset_by_location:
            return int((date - offset_by_location[location]) / pd.Timedelta(days=1))

    tdf['days since 100 cases - ' + group_col] = tdf.apply(
        shift_dates,
        offset_by_location=day_location_reached_100,
        axis='columns'
    )
    tdf['days since 100 active - ' + group_col] = tdf.apply(
        shift_dates,
        offset_by_location=day_location_reached_100_active,
        axis='columns'
    )
    
    tdf['days since 1 deaths - ' + group_col] = tdf.apply(
        shift_dates,
        offset_by_location=day_location_reached_1_deaths,
        axis='columns'
    )
    tdf['days since 10 deaths - ' + group_col] = tdf.apply(
        shift_dates,
        offset_by_location=day_location_reached_10_deaths,
        axis='columns'
    )
    tdf['days since 100 deaths - ' + group_col] = tdf.apply(
        shift_dates,
        offset_by_location=day_location_reached_100_deaths,
        axis='columns'
    )
#     tdf['days since 1 case/100k people - ' + group_col] = tdf.apply(
#         shift_dates,
#         offset_by_location=day_location_reached_1_per_100k,
#         axis='columns'
#     )
    return tdf


# In[ ]:


geopy.geocoders.options.default_timeout = 30
locator = Nominatim(user_agent="mesur.io")
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
revgeocode = RateLimiter(locator.reverse, min_delay_seconds=1)


# In[ ]:


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


# In[ ]:


combined = combined_csv.copy()


# In[ ]:


combined.columns


# In[ ]:


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


# In[ ]:


combined['Last Update'] = pd.to_datetime(combined['Last Update']).dt.round(freq = 'D')
combined = combined.sort_values('Last Update').reset_index(drop=True)


# In[ ]:


combined['Geo_Input'] = combined['Province/State']+', '+combined['Country/Region'] 


# In[ ]:


non_located = combined[combined['Latitude'].isna()]
non_located = non_located[non_located['Province/State'] != 'Cruise Ship']


# In[ ]:


geo_inputs = non_located['Geo_Input'].unique()
combined['Location_Key_Raw'] = combined.apply(lambda x: (x.Latitude, x.Longitude), axis = 1)
#for testing you may want to trim this down a bit
#geo_inputs = geo_inputs[:10]


# In[ ]:


def geocode_jh():
    print('Geocoding for: ', len(geo_inputs), 'locations')
    #use progress_apply() for interactive progress
    d = dict(zip(geo_inputs, pd.Series(geo_inputs).apply(geocode).apply(lambda x: (x.latitude if pd.notnull(x.latitude) else x.latitude, 
                                                                                   x.longitude if pd.notnull(x.longitude) else x.longitude) if pd.notnull(x) else x)
                )
            )
    pickle.dump(d, open('./reference/geolod_dict.pickle', 'wb'))


# In[ ]:


#geocode_jh()


# In[ ]:


d = pickle.load(open('./reference/geolod_dict.pickle', 'rb'))


# In[ ]:


combined['Location_Key'] = combined['Geo_Input'].map(d)
combined['Location_Key'] = combined['Location_Key'].fillna(combined['Location_Key_Raw'])
combined['Latitude'] = combined.loc[combined['Latitude'].isna(), 'Location_Key'].apply(lambda x: x[0])
combined['Longitude'] = combined.loc[combined['Longitude'].isna(), 'Location_Key'].apply(lambda x: x[1])


# In[ ]:


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


# In[ ]:


combined = combined.fillna(0)


# In[ ]:


# do a little reording and subselection
combined_csv = combined[['Last Update','Latitude','Longitude','Country/Region','Province/State','FIPS','Admin2',
                         'Confirmed','Deaths','Recovered','UnknownActive', 'Active',
                         'Day','DayLoc','DayCountry','DayCountryProvince']]
combined_csv = combined_csv.sort_values(['Last Update','Latitude','Longitude','Country/Region','Province/State'])
#combined_csv


# In[ ]:


def get_state_country_jh(row):
    location_segments = [
        row['Province/State'], row['Country/Region']
    ]
    cleaned_location_segments = [
        segment
        for segment in location_segments
        if type(segment) is str
    ]
    return ', '.join(cleaned_location_segments)

combined_csv['State and Country'] = combined_csv.apply(get_state_country_jh, axis='columns')


# In[ ]:


combined_csv = calc_timeseries_by_group(combined_csv, 'Country/Region')
combined_csv = calc_timeseries_by_group(combined_csv, 'State and Country')


# In[ ]:


combined_csv.to_csv(output_dir + "combined.csv", index=False, encoding='utf-8-sig')


# In[ ]:


web_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases.csv')
web_cases_state = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_state.csv')
web_cases_country = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
web_cases_time = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv')

web_cases.to_csv(output_dir + "web_cases.csv", index=False, encoding='utf-8-sig')
web_cases_state.to_csv(output_dir + "web_cases_state.csv", index=False, encoding='utf-8-sig')
web_cases_country.to_csv(output_dir + "web_cases_country.csv", index=False, encoding='utf-8-sig')
web_cases_time.to_csv(output_dir + "web_cases_time.csv", index=False, encoding='utf-8-sig')


# In[ ]:





# In[ ]:


df = combined_csv.copy()
df['Last Update'] = pd.to_datetime(df['Last Update']).dt.round(freq = 'D')

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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

ts_deaths['Date'] = pd.to_datetime(ts_deaths['Date']).dt.round(freq = 'D')
ts_confirmed['Date'] = pd.to_datetime(ts_confirmed['Date']).dt.round(freq = 'D')
ts_recovered['Date'] = pd.to_datetime(ts_recovered['Date']).dt.round(freq = 'D')

#and concat into one nice set
covid_19_ts = ts_deaths.copy()
covid_19_ts = covid_19_ts.append(ts_recovered)
covid_19_ts = covid_19_ts.append(ts_confirmed)
covid_19_ts = covid_19_ts.sort_values(['Country/Region', 'Province/State', 'Date']).reset_index(drop=True)

#now drop 0 values
covid_19_ts = covid_19_ts[covid_19_ts['Observation'] != 0]


# In[ ]:


overallDf.to_csv(output_dir + "covid_19_by_date_and_country.csv", index=False, encoding='utf-8-sig')
covid_19_ts.to_csv(output_dir + "covid_19_ts.csv", index=False, encoding='utf-8-sig')


# In[ ]:


#display for debug
#display(covid_19_ts)


# In[ ]:


# sourcing from CDS here: https://coronadatascraper.com/#home
# we really like these guys, but would recommend that you fork and spin up your own scraper set
# set the following url to your own source
scraper = pd.read_csv('https://coronadatascraper.com/timeseries.csv')
scraper['date'] = pd.to_datetime(scraper['date']).dt.round(freq = 'D')
scraper.to_csv(output_dir+'scraper_raw.csv', index=False, encoding='utf-8-sig')


# In[ ]:


scraper['cases'] = scraper['cases'].fillna(0)
scraper['recovered'] = scraper['recovered'].fillna(0)
scraper['active'] = scraper['active'].fillna(0)
scraper['tested'] = scraper['tested'].fillna(0)
scraper['growthFactor'] = scraper['growthFactor'].fillna(0)


# In[ ]:


scraper['Geo_Input'] = scraper['state']+', '+scraper['country'] 
non_located = scraper[scraper['lat'].isna()]
geo_inputs = non_located['Geo_Input'].dropna().unique()
scraper['Location_Key_Raw'] = scraper.apply(lambda x: (x.lat, x.long), axis = 1)
scraper['Location_Key'] = scraper.apply(lambda x: (x.lat, x.long), axis = 1)


# In[ ]:


def geocode_scraper():
    print('Geocoding for: ', len(geo_inputs), 'locations')
    d = dict(zip(geo_inputs, pd.Series(geo_inputs).apply(geocode).apply(lambda x: (x.latitude if pd.notnull(x.latitude) else x.latitude, 
                                                                                   x.longitude if pd.notnull(x.longitude) else x.longitude) if pd.notnull(x) else x)
                )
            )
    pickle.dump(d, open('./reference/geoloc_dict_scraper.pickle', 'wb'))


# In[ ]:


#geocode_scraper()


# In[ ]:


d = pickle.load(open('./reference/geoloc_dict_scraper.pickle','rb'))


# In[ ]:


scraper['Location_Key'] = scraper['Geo_Input'].map(d)
scraper['Location_Key'] = scraper['Location_Key'].fillna(scraper['Location_Key_Raw'])
scraper.loc[scraper['lat'].isna(), 'lat'] = scraper.loc[scraper['lat'].isna(), 'Location_Key'].apply(lambda x: (x[0] if pd.notnull(x[0]) else x[0]) if pd.notnull(x) else x)
scraper.loc[scraper['lat'].isna(), 'long'] = scraper.loc[scraper['lat'].isna(), 'Location_Key'].apply(lambda x: (x[1] if pd.notnull(x[1]) else x[1]) if pd.notnull(x) else x)


# In[ ]:


def revgeocode_scraper():
    rev_set = scraper[['lat', 'long']].dropna().drop_duplicates()
    print('Reverse geocoding for', rev_set.shape[0],'locations')
    rev_list = rev_set['lat'].astype(str) + ', ' + rev_set['long'].astype(str)
    r = rev_list.values
    d = dict(zip(rev_list, pd.Series(r).apply(revgeocode).apply(lambda x: x if pd.notnull(x) else x)))
    pickle.dump(d, open('./reference/geoloc_rev_dict_scraper.pickle', 'wb'))


# In[ ]:


# revgeocode_scraper()./reference/


# In[ ]:


country_codes = pd.read_csv('./reference/country_region_mappings.csv').set_index('alpha-3')['name']
country_codes.name = 'country'
# do some cleanup on this stuff
country_codes.update(pd.Series({
    'USA': 'USA',
    'GBR': 'UK',
    'KOR': 'South Korea',
}))
# now let's do some display friendly naming thatnks to Jason Curtis
def get_combined_location(row):
    location_segments = [
        row['city'], row['county'], row['state'], row['country']
    ]
    cleaned_location_segments = [
        segment
        for segment in location_segments
        if type(segment) is str
    ]
    return ', '.join(cleaned_location_segments)

cleaned_timeseries = (
    scraper.rename(
        {
            'country': 'country_code'
        },
        axis='columns'
    ).join(country_codes, 'country_code')
)

cleaned_timeseries['location'] = cleaned_timeseries.apply(get_combined_location, axis='columns')
#cleaned_timeseries


# In[ ]:


cleaned_timeseries.to_csv(output_dir+'scraper_cleaned.csv', index=False, encoding='utf-8-sig')


# In[ ]:


# now reshape and rename for backwards compat
cleaned_timeseries['date'] = pd.to_datetime(cleaned_timeseries['date']).dt.round(freq = 'D')
cleaned_timeseries['Last Update'] = cleaned_timeseries['date']
scraper_df = cleaned_timeseries[['Last Update', 'date', 'lat', 'long', 'location', 'city', 'county', 'state', 'country', 'population', 'active', 'cases', 'deaths', 'recovered', 'tested', 'growthFactor']].copy()
scraper_df.columns = ['Last Update', 'Date', 'Latitude', 'Longitude', 'Location', 'City', 'County', 'State', 'Country', 'Population', 'Active', 'Confirmed', 'Deaths', 'Recovered', 'Tested', 'Growth Factor']


# In[ ]:


def get_state_country(row):
    location_segments = [
        row['State'], row['Country']
    ]
    cleaned_location_segments = [
        segment
        for segment in location_segments
        if type(segment) is str
    ]
    return ', '.join(cleaned_location_segments)

scraper_df['State and Country'] = scraper_df.apply(get_state_country, axis='columns')


# In[ ]:


# good, looks like only dupes are due to NaN on lat/lon - this can be corrected with better reverse geocoding
#scraper_df[scraper_df[['Last Update', 'Latitude', 'Longitude']].duplicated()]
#scraper_df


# In[ ]:


scraper_df['Confirmed Death Rate'] = scraper_df['Confirmed'] / scraper_df['Deaths']
scraper_df['Confirmed per 100k capita'] = scraper_df['Confirmed'] / scraper_df['Population'] * 1e5


# In[ ]:


# this is once again some great work from Jason Curtis
scraper_df['Date'] = scraper_df['Date'].apply(lambda date: date.strftime('%Y-%m-%d'))


# In[ ]:





# In[ ]:


scraper_df = calc_timeseries_by_group(scraper_df, 'Country')
scraper_df = calc_timeseries_by_group(scraper_df, 'State and Country')
scraper_df = calc_timeseries_by_group(scraper_df, 'Location')


# In[ ]:


scraper_df = scraper_df.fillna(0)


# In[ ]:


scraper_df.to_csv(output_dir+'scraper.csv', index=False, encoding='utf-8-sig')


# In[ ]:


# pd.set_option('display.max_columns', None)
# scraper_df


# In[ ]:




