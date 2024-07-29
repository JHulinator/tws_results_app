import calendar
from datetime import datetime
import pandas as pd
import glob
import os
from app import get_miles_from_start

GAUGES = pd.read_csv('data/gauge_list.csv')

# Make a list of the dates
start_days = [] # This will be a list of strings containing the second Sat in June for years from 1963 to now
today = datetime.now().year
cal = calendar.Calendar(6)

for year in range(1963, today+1, 1):
    # print(calendar.TextCalendar(6).prmonth(year, 6))
    start_day = cal.monthdayscalendar(year, 6)[1][-1]
    # print(cal.monthdayscalendar(year, 6)[1][-1])
    start_days.append(f'{year}-06-{start_day}')

# Get all the flow data
l = []
path = r'data\usgs'
all_files = glob.glob(os.path.join(path, "*.tsv"))

for fl in all_files:
    df = pd.read_csv(fl, sep='\t', header=31)
    keep_cols = [col for col in df.columns if col == 'site_no' or col == 'datetime' or ('_00065_00003' in col and not '_cd'in col) or ('_00060_00003' in col and not '_cd' in col)]
    df = df.loc[df['datetime'].isin(start_days), keep_cols]
    df.rename(columns=lambda col: 'gage_height' if '00065' in col else ('discharge' if '00060' in col else col), inplace=True)
    l.append(df)

print(pd.concat(l, axis=0, ignore_index=True))
    

# flow_data = pd.concat((pd.read_csv(f, sep='\t', header=31) for f in all_files), ignore_index=True)
# # Filter irrelevant data
# flow_data = flow_data.loc[:, [col for col in flow_data.columns if col == 'site_no' or col == 'datetime' or ('_00065_00003' in col and not '_cd'in col) or ('_00060_00003' in col and not '_cd' in col)]]
# print(flow_data)