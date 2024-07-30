import calendar
from datetime import datetime
import pandas as pd
import glob
import os
from app import get_miles_from_start

GAUGES = pd.read_csv('data/gauge_list.csv', index_col='gauge_number')
# convert lat lon format
GAUGES['lat'] = GAUGES['lat'].str.replace('"', '').str.split(r'[°\']', regex=True).apply(lambda lat: int(lat[0]) + int(lat[1])/60 + float(lat[2])/3600)
GAUGES['lon'] = GAUGES['lon'].str.replace('"', '').str.split(r'[°\']', regex=True).apply(lambda lon: -1*(int(lon[0]) + int(lon[1])/60 + float(lon[2])/3600))

# Add millage column
GAUGES['milage'] = GAUGES.apply(lambda x: get_miles_from_start(latitude=x['lat'], longitude=x['lon']), axis=1)

# Make a list of the dates
start_days = [] # This will be a list of strings containing the second Sat in June for years from 1963 to now
today = datetime.now().year
cal = calendar.Calendar(6)

for year in range(1963, today+1, 1):
    start_day = cal.monthdayscalendar(year, 6)[1][-1]
    start_days.append(f'{year}-06-{start_day}')

# Get all the flow data
l = []
path = r'data\usgs'
all_files = glob.glob(os.path.join(path, "*.tsv"))

for fl in all_files:
    df = pd.read_csv(fl, sep='\t', header=31)
    keep_cols = [col for col in df.columns if col == 'site_no' or col == 'datetime' or ('_00065_00003' in col and not '_cd'in col) or ('_00060_00003' in col and not '_cd' in col)]
    df = df.loc[df['datetime'].isin(start_days), keep_cols]
    df.rename(columns=lambda col: 'Gage height' if '00065' in col else ('Discharge' if '00060' in col else col), inplace=True)
    df['year'] = df['datetime'].apply(lambda x: x.split('-')[0])
    l.append(df)

flow_data = pd.concat(l, axis=0, ignore_index=True)
# convert flow data to integer
flow_data['site_no'] = flow_data['site_no'].astype(int)
flow_data = flow_data.melt(id_vars=['site_no', 'datetime', 'year'], value_vars=['Gage height', 'Discharge']).dropna(subset='value')
    
# Add column of the name
flow_data['site_name'] = flow_data['site_no'].apply(lambda num: GAUGES.loc[num, 'name'])

# # Add a columns for lat and lon
flow_data['lat'] = flow_data['site_no'].apply(lambda num: GAUGES.loc[num, 'lat'])
flow_data['lon'] = flow_data['site_no'].apply(lambda num: GAUGES.loc[num, 'lon'])

# Add a column for the millage
flow_data['milage'] = flow_data['site_no'].apply(lambda num: GAUGES.loc[num, 'milage'])

# Add UOM column
flow_data['UOM'] = flow_data['variable'].apply(lambda x: 'ft' if x == 'Gage height' else 'ft^3/s')

flow_data.to_csv('flow_data.csv')
print(flow_data)

