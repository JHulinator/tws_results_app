# region Imports ------------------------------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore') # , category=FutureWarning
from dash import Dash, html, dash_table, dcc, Input, Output, State, callback, Patch, clientside_callback, ctx, set_props
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import pandas as pd
import numpy as np

import gpxpy
import gpxpy.gpx

from math import radians, cos, sin, asin, sqrt
import os
import re
from difflib import SequenceMatcher
from os.path import exists
from typing import List
import ast
# endregion imports ---------------------------------------------------------------------------------------------------


# region Globals  -----------------------------------------------------------------------------------------------------
# Global Constants
TWS_ROUTE = gpxpy.parse(open('assets/tws_race_route.gpx', 'r'))
TWS_TOTAL_MILES: float
TWS_CHECKPOINTS: pd.DataFrame
DEBUG = False
CLASS_LIST = list(pd.read_csv('assets/class_list.csv', sep=',')['class'])


DISP_TYP_DICT = {
    'Time of day':'time_of_day',
    'Total time':'str_hours',
    'Split time':'str_split_time',
    'Speed':'Split Speed',
}


# Global Variables
# stylesheet with the .dbc class to style  dcc, DataTable and AG Grid components with a Bootstrap theme
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

external_stylesheets = [dbc.themes.FLATLY, dbc.icons.BOOTSTRAP, dbc_css]
app = Dash(external_stylesheets=external_stylesheets)
server = app.server
year = 2024
years = os.listdir('assets/split_data/')
data: pd.DataFrame
# endregion globals ---------------------------------------------------------------------------------------------------


# region helper methods -----------------------------------------------------------------------------------------------
def str_to_hours(str_series:pd.Series) -> pd.Series:
    df = pd.DataFrame.from_dict(dict(zip(str_series.str.split(':').index, str_series.str.split(':').values))).T
    return_series = pd.to_numeric(df.iloc[:,0], errors='coerce').fillna(0) + pd.to_numeric(df.iloc[:,1], errors='coerce').fillna(0)/60 + pd.to_numeric(df.iloc[:,2], errors='coerce').fillna(0)/3600
    return return_series



def haversine(lon1: float, lat1 :float, lon2 :float, lat2 :float) -> float:
    """
    Calculate the great circle distance in miles between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in miles. Use 6371 for kilometers. Determines return value units.
    return c * r


def get_distance_between(point1:gpxpy.gpx.GPXTrackPoint, point2:gpxpy.gpx.GPXTrackPoint) -> float:
    #Returns the distance in miles between two GPX TrackPoints
    return haversine(point1.longitude, point1.latitude, point2.longitude, point2.latitude)


def get_track_len(track:gpxpy.gpx.GPXTrack) -> float:
    track_miles = 0.0
    for seg in track.segments:
        # get length of current segment
        seg_miles = 0.0
        for n, pt in enumerate(seg.points):
            # Get the point length
            pt_miles = 0.0
            # Skip the last point
            if n != len(seg.points)-1:
                pt_miles = get_distance_between(pt, seg.points[n+1])
            seg_miles += pt_miles
        
        # add segment length to track length
        track_miles += seg_miles
        pass
    return track_miles


def get_milage_to_finish(latitude:float, longitude:float) -> float:
    # This method calculates the milage to the TWS finish of any point
    nearest_index = 0
    nearest_dist = None

    # Find the nearest route point
    for n, pt in enumerate(TWS_ROUTE.tracks[0].segments[0].points):
        # Calculate distance
        this_dist = haversine(pt.longitude, pt.latitude, longitude, latitude)

        # See if less then nearest
        if (nearest_dist==None or nearest_dist > this_dist):
           nearest_dist = this_dist
           nearest_index = n 
    
    nearest_point = TWS_ROUTE.tracks[0].segments[0].points[nearest_index]
    
    # Find the distance to that point
    distance_to_finish = haversine(nearest_point.longitude, nearest_point.latitude, longitude, latitude)


    # Find the remaining distance from the nearliest point to finish
    for n, pt in enumerate(TWS_ROUTE.tracks[0].segments[0].points):
        if n >= nearest_index:
            # Get the point length
            pt_miles = 0.0

            # Skip the last point
            if n != len(TWS_ROUTE.tracks[0].segments[0].points)-1:
                pt_miles = get_distance_between(pt, TWS_ROUTE.tracks[0].segments[0].points[n+1])

            distance_to_finish += pt_miles
            
    return distance_to_finish


def get_speed(point1:gpxpy.gpx.GPXTrackPoint, point2:gpxpy.gpx.GPXTrackPoint) -> float:
    if (point1 is None) or (point2 is None):
        speed = 0
    else:
        distance = abs(get_milage_to_finish(point1.latitude, point1.longitude) - get_milage_to_finish(point2.latitude, point2.longitude))
        delta_t = abs((point1.time - point2.time).total_seconds() / 60 / 60)
        if delta_t == 0:
            speed = 0
        else:
            speed = distance / delta_t
    return speed


def get_miles_from_start(latitude:float, longitude:float) -> float:
    to_finish = get_milage_to_finish(latitude=latitude, longitude=longitude)
    return TWS_TOTAL_MILES - to_finish


# Calculate the total miles of the TWS course in the tws_race_route.gpx file
TWS_TOTAL_MILES = get_track_len(TWS_ROUTE.tracks[0])

# Read The list of Checkpoints
TWS_CHECKPOINTS = pd.read_csv('assets/tws_checkpoint_list.csv', sep=',', index_col=0)

# Add a milage for each checkpoint
TWS_CHECKPOINTS['Milage'] = 0.0

for row in TWS_CHECKPOINTS.axes[0]:
    TWS_CHECKPOINTS['Milage'].loc[row] = get_miles_from_start(TWS_CHECKPOINTS['latitude'].loc[row], TWS_CHECKPOINTS['longitude'].loc[row])

# Sort ascending milage
TWS_CHECKPOINTS = TWS_CHECKPOINTS.sort_values('Milage')

def find_class(str_class:str) -> str:
    '''----------------------------------------------------------------------------------------------------------------
    This method interoperates the many class names recorded in the 'Recognition' column and returns a string matching
    one of the 14 classed recognized by TWS per the webpage as of 7/4/24
    ----------------------------------------------------------------------------------------------------------------'''
    # remove and insignificant characters
    str_class = str(str_class)
    str_class = re.sub(r'\W|^\d|[3-9]', '', str_class.replace('1st ', '').replace('2nd ','').replace('3rd ', '').replace('th ', '').replace('0', '')).lower()
    str_class = re.sub(r'^\d', '', str_class)
    str_class = str_class.replace('masters', '').replace('adultyouth', '').replace('adultchild', '').replace('woman', 'women')
    
    # Define search terms
    women = 'women' in str_class
    unlimited = 'unlimited' in str_class
    tandem = 'tandem' in str_class
    novice = 'novice' in str_class
    solo = 'solo' in str_class
    uscac2 = 'uscac2' in str_class
    uscac1 = 'uscac1' in str_class
    standard = 'standard' in str_class
    mixed = 'mixed' in str_class
    
    if not women and unlimited and not tandem and not novice and not solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[0] # Unlimited
    elif women and unlimited and not tandem and not novice and not solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[1] # Women's Unlimited
    elif not women and unlimited and tandem and not novice and not solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[2] # Tandem Unlimited
    elif women and unlimited and tandem and not novice and not solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[3] # Women's Tandem Unlimited
    elif not women and unlimited and not tandem and not novice and solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[4] # Men's Solo Unlimited
    elif women and unlimited and not tandem and not novice and solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[5] # Women's Solo Unlimited
    elif not women and not unlimited and not tandem and not novice and not solo and uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[6] # USCA C2
    elif not women and not unlimited and not tandem and not novice and not solo and not uscac2 and uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[7] # Men's C1
    elif women and not unlimited and not tandem and not novice and not solo and not uscac2 and uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[8] # Women's C1
    elif not women and not unlimited and not tandem and not novice and not solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[9] # Aluminum
    elif not women and not unlimited and not tandem and not novice and not solo and not uscac2 and not uscac1 and standard and not mixed:
        return_class = CLASS_LIST[10] # Standard
    elif not women and not unlimited and not tandem and novice and not solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[11] # Novice
    elif women and not unlimited and not tandem and novice and not solo and not uscac2 and not uscac1 and not standard and not mixed:
        return_class = CLASS_LIST[12] # Women's Novice
    elif not women and not unlimited and not tandem and novice and not solo and not uscac2 and not uscac1 and not standard and mixed:
        return_class = CLASS_LIST[13] # Mixed
    else:
        # find the class that is the most similar to the string
        similarity_ratio = 0
        return_class = None
        for cl in CLASS_LIST:
            if SequenceMatcher(None, str_class, cl.lower()).ratio() > similarity_ratio:
                similarity_ratio = SequenceMatcher(None, str_class, cl.lower()).ratio()
                return_class = cl

    #print(f'Given:{str_class}\nReturned: {return_class}\n\n')
    return return_class

def find_gender(str_class:str) -> str:
    df_class = pd.read_csv('assets/class_list.csv', sep=',', index_col='class')['gender']
    return str(df_class.loc[str_class])

def find_hull_ln(str_class:str) -> str:
    df_class = pd.read_csv('assets/class_list.csv', sep=',', index_col='class')['hull lenght restriction']
    return str(df_class.loc[str_class])

def find_hull_width(str_class:str) -> str:
    df_class = pd.read_csv('assets/class_list.csv', sep=',', index_col='class')['hull width restriction']
    return str(df_class.loc[str_class])

def is_rudder(str_class:str) -> bool:
    df_class = pd.read_csv('assets/class_list.csv', sep=',', index_col='class')['rudder restriction']
    return not bool(df_class.loc[str_class])

def is_double_blade(str_class:str) -> bool:
    df_class = pd.read_csv('assets/class_list.csv', sep=',', index_col='class')['double blade restriction']
    return not bool(df_class.loc[str_class])

def is_masters(str_recognition:str) -> bool:
    str_class = str(str_recognition)
    str_class = re.sub(r'\W|^\d|[3-9]', '', str_class.replace('1st ', '').replace('2nd ','').replace('3rd ', '').replace('th ', '').replace('0', '')).lower()
    str_class = re.sub(r'^\d', '', str_class)
    return ('masters' in str_class)

def is_adult_youth(str_recognition:str) -> bool:
    str_class = str(str_recognition)
    str_class = re.sub(r'\W|^\d|[3-9]', '', str_class.replace('1st ', '').replace('2nd ','').replace('3rd ', '').replace('th ', '').replace('0', '')).lower()
    str_class = re.sub(r'^\d', '', str_class)
    return ('adult' in str_class)


def name_formatter(names:str) -> str:
    '''
    This function takes the values in the Team Members column and outputs a formatted string containing first initials and las names
    '''
    names = names.split(';') # Make a list of all team members names
    name_str=''
    for name in names:
        if 'TC' not in name and len(name) > 2:
            name = re.sub(r'^\s', '', name) # Remove any leading spaces
            last = name.split(' ')[1] # Get the last name
            name_str += f'{name[0]} {last} | '
    name_str = name_str[:-3] # Remove the last separator
    return name_str

def cont_competitors(names:str) -> int:
    names = names.split(';') # Make a list of all team members names
    count = 0
    for name in names:
        if 'TC' not in name and len(name) > 2:
            count += 1
    return count

def get_raw_data(year: int) -> pd.DataFrame:
    # print('Enter: get_raw_data ---------------\n')
    # This method reads the CSV split spreadsheet data as downloaded from https://www.texaswatersafari.org/

    # Read the data CSV file
    file_str = 'assets/split_data/' + str(year) + '/' + str(year) + '.csv'
    df = pd.read_csv(file_str, sep=',', header=6)

    # Rename some columns. This is needed because the headers titles in the CSV do not align with the values (Why TWS?!?!).
    # Start with the colum named 'Staples' and iterate through the data columns
    for n in range(df.columns.get_loc('Staples'), len(df.columns)-1, 2): 
        df = df.rename(columns={df.columns[n]:'Unnamed', df.columns[n+1]:df.columns[n]})

    # The CSV file has a lot of empty cells (for excel hell formatting). We are now going to get rid of these
    # meaningless cells.
    df.drop(df.columns[df.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True) # This Drops all the unnamed columns
    df.iloc[:, 4:] = df.iloc[:, 4:].shift(-1) # This shifts the time data up so that total cumulative time is aligned with team info line. This will be the only data that is kept
    df = df.loc[df['Overall Place'].notna()]# df[df[df.columns[0]].notna()] # This drops all the rows that don't have info in the first column
    df.reset_index(drop=True, inplace=True) # This resets the index of the rows that are left


    # The Time data columns are of type object, so hear we convert them to timedelta type
    # df.iloc[:, 4:] = df.iloc[:, 4:].astype('string') # First convirt them to type = string
    for col in range(4, len(df.columns)): # Iterate through all time data columns
        # Change each time column form type string to timedelta
        df.iloc[:, col] = pd.to_timedelta(df.iloc[:,col], errors='coerce')


    # Add columns containing the boat class
    df['Class'] = df['Recognition'].apply(find_class)
    df['Gender'] = df['Class'].apply(find_gender)
    df['Max Boat Len'] = df['Class'].apply(find_hull_ln)
    df['Min Boat Width'] = df['Class'].apply(find_hull_width)
    df['Rudder'] = df['Class'].apply(is_rudder)
    df['Double Blade'] = df['Class'].apply(is_double_blade)

    # Add column for any special recognitions
    df['Masters'] = df['Recognition'].apply(is_masters)
    df['Adult Youth'] = df['Recognition'].apply(is_adult_youth)
    

    #Split out competitors and team captions
    df[['Competitors', 'Team Captions']] = pd.DataFrame(df['Team Members'].astype(str).str.split(pat='TC ', n=1 , regex=True).to_list(), index=df.index)

    # Convert str to list of str
    df['Competitors'] = df['Competitors'].str.replace('\n{2,}', '').str.split('\r\n', regex=True)
    df['Competitors'] = df['Competitors'].apply(lambda a: list(filter(None, list(a))))
    df['Team Captions'] = df['Team Captions'].str.replace('TC |\n{2,}', '').str.split('\r\n',regex=True)

    # Add column for competitor count
    df['Competitor count'] = df['Competitors'].apply(lambda x: len(x))

    # Convert Boat Number to Int
    df['Boat #'] = df['Boat #'].astype(int)

    # Format newlines for plotly to recognize
    df['Team Members'] = df['Team Members'].str.replace('\n\n', '\n')
    df['Team Members'] = df['Team Members'].str.replace('\n', '; ')

    # Add column for class place
    df['Class Place'] = df['Recognition'].astype(str).str.split(' ').apply(lambda a: a[0])
    df['Class Place'] = df['Class Place'].str.replace(r'\D', '', regex=True)


    # Rearrange the wide split data into long split data
    df = pd.melt(df,
    id_vars=['Overall Place', 'Recognition', 'Team Members', 'Boat #', 'Class', 'Gender', 'Max Boat Len',
       'Min Boat Width', 'Rudder', 'Double Blade', 'Masters', 'Adult Youth',
       'Competitors', 'Team Captions', 'Competitor count', 'Class Place'],
       var_name='Split Name',
       value_name='Split Time'
       )
    
    # Add year column
    df['year'] = year

    # Change Cuero 72 to Cheapside
    df['Split Name'].loc[df['Split Name'] == 'Cuero 766'] = 'Cheapside'
              
    # Add a column with the string formatted split_time
    df['str_split_time'] = df['Split Time'].apply(lambda x: f'{divmod(x.seconds, 3600)[0] + 24*x.days}:{divmod(divmod(x.seconds, 3600)[1], 60)[0]}:{divmod(divmod(x.seconds, 3600)[1], 60)[1]}')

    # Add a column with the milage at each split
    df['Milage'] = df['Split Name'].apply(lambda a: TWS_CHECKPOINTS.loc[a, 'Milage'])

    # Add a column with the split milage
    this_year_cp_list = TWS_CHECKPOINTS.loc[df['Split Name'].unique()]
    this_year_cp_list['Split Milage'] = this_year_cp_list['Milage'].diff().fillna(this_year_cp_list['Milage'])
    df['Split Milage'] = df['Split Name'].apply(lambda a: this_year_cp_list.loc[a, 'Split Milage'])
    

    # Add a column with the total hours
    df['Hours'] = df.apply(lambda x: x['Split Time'] + df['Split Time'].loc[(df['Boat #'] == x['Boat #']) & (df['Milage'] < x['Milage'])].sum(), axis=1)

    # Add a column with the string formatted total_time
    df['str_hours'] = df['Hours'].apply(lambda x: f'{divmod(x.seconds, 3600)[0] + 24*x.days}:{divmod(divmod(x.seconds, 3600)[1], 60)[0]}:{divmod(divmod(x.seconds, 3600)[1], 60)[1]}')

    # Add column with the hours as float
    df['float_hours'] = df['Hours'].apply(lambda x: 24*x.days + x.seconds / 3600)

    # Add a column with the string formatted time_of_day
    df.dropna(axis=0, subset='Split Time', inplace=True)
    df['datetime_2000_1_1'] = df['Hours'].apply(lambda x: pd.to_datetime('2000-1-1 09:00') + x) # df['Hours'].apply(lambda x: f'{day_dict[1+x.days+1*(divmod(x.seconds, 3600)[0]>15)]} {9+divmod(x.seconds, 3600)[0]-24*(divmod(x.seconds, 3600)[0]>15)}:{divmod(divmod(x.seconds, 3600)[1], 60)[0]}')
    
    df['time_of_day'] = df['datetime_2000_1_1'].dt.strftime('%a %H:%M') #dt.apply(lambda x: f'{day_dict[x.day]} {x.time.hour}:{x.time.minute}')

    # Add a column with the split speed
    df['Split Speed'] = df.apply(lambda x: x['Split Milage'] / (x['Split Time'].seconds / 3600), axis=1)

    # Add a column for the finish time
    df['Finish time'] = df.apply(lambda z: df['Hours'].loc[df['Boat #'] == z['Boat #']].max().seconds / 3600 + df['Hours'].loc[df['Boat #'] == z['Boat #']].max().days * 24, axis=1)
    
    # Get a string formatted team name
    df['Team Name'] = df['Team Members'].apply(name_formatter)

    # Get compeditor count
    df['Competitor count'] = df['Team Members'].apply(cont_competitors)

    # print('Exit: get_raw_data ---------------\n')
    return df


def get_all_raw_data() -> pd.DataFrame:
    print('Enter get all raw data ------------------')
    df = pd.DataFrame()
    for yr in years:
        yr_df = get_raw_data(yr)
        df = pd.concat([df, yr_df], ignore_index=True, sort=False)
    print('Exit get all raw data -------------------')
    return df


def filter_data(year_filter:List[int]=years, class_filter:List[str]=CLASS_LIST,
                pos_filter:int=0, cl_pos_filter:int=0, gender_filter:List[str]=['Undefined', 'Male', 'Female', 'Mixed'],
                count_filter:List[int]=[1,2,3,4,5,6], rudder_filter:bool=False, blade_filter:bool=False, 
                masters_filter:bool=False, adult_youth_filter:bool=False, time_filter:List[float]=[0, 100]) -> pd.DataFrame:
    
    year_filter = list(map(int, year_filter)) # Convert to int if not already
    time_filter = list(map(float, time_filter))
    if pos_filter == 0: pos_filter = np.inf
    if cl_pos_filter == 0: cl_pos_filter = np.inf
    df = full_df
    filtered = df.loc[
                (df['year'].isin(year_filter)) & 
                (df['Class'].isin(class_filter)) & 
                (pd.to_numeric(df['Overall Place'], errors='coerce') <= pos_filter) &
                (pd.to_numeric(df['Class Place'], errors='coerce') <= cl_pos_filter) &
                (df['Gender'].isin(gender_filter)) & 
                (df['Competitor count'].isin(count_filter)) &
                (~(rudder_filter & df['Rudder'])) &
                (~(blade_filter & df['Double Blade'])) & 
                (~(masters_filter & (masters_filter ^ df['Masters']))) & 
                (~(adult_youth_filter & (adult_youth_filter ^ df['Adult Youth']))) &
                ((df['Finish time'] >= time_filter[0]) & (df['Finish time'] <= time_filter[1]))
                ]
    
    return filtered

def update_year(id:str, selected_yrs, multi_value) -> List:
    last5years = [years[i] for i in np.argsort(years)[-5:]]
    last10years = [years[i] for i in np.argsort(years)[-10:]]
    if id == 'year_filter':
        if selected_yrs == [max(years)]:
            return_val= 1, selected_yrs
        elif len(selected_yrs) == 5 and set(selected_yrs) == set(last5years):
            return_val= 5, selected_yrs
        elif len(selected_yrs) == 10 and set(selected_yrs) == set(last10years):
            return_val= 10, selected_yrs
        elif set(selected_yrs) == set(years):
            return_val= 0, selected_yrs
        else:
            return_val= -1, selected_yrs
    elif id == 'year-multi-select':
        if multi_value == 1:
            return_val= multi_value, [max(years)]
        elif multi_value == 5:
            return_val= multi_value, last5years
        elif multi_value == 10:
            return_val= multi_value, last10years
        elif multi_value == 0:
            return_val= multi_value, years
    else:
        return_val= multi_value, selected_yrs
    return return_val

def update_class(id:str, class_filter, class_filter_an) -> List:
    if id == 'class_filter':
        if set(class_filter) == set(CLASS_LIST):
            class_filter_an = 1
        elif class_filter == []:
            class_filter_an = 0
        else:
            class_filter_an = None
    elif id == 'class_filter_an':
        if class_filter_an == 0:
            class_filter = []
        elif class_filter_an == 1:
            class_filter = CLASS_LIST

    return class_filter, class_filter_an

def update_gender(id:str, gender_filter, gender_filter_an) -> List:
    gender_list = ['Undefined', 'Male', 'Female', 'Mixed']
    if id == 'gender_filter':
        if set(gender_list) == set(gender_filter):
            gender_filter_an = 1
        elif gender_filter == []:
            gender_filter_an = 0
        else:
            gender_filter_an = None
    elif id == 'gender_filter_an':
        if gender_filter_an == 0:
            gender_filter = []
        elif gender_filter_an == 1:
            gender_filter = gender_list
    return gender_filter, gender_filter_an

def update_count(id:str, count_filter, count_filter_an) -> List:
    allowable_count = [1,2,3,4,5,6]
    if id == 'count_filter':
        if set(count_filter) == set(allowable_count):
            count_filter_an = 1
        elif count_filter == []:
            count_filter_an = 0
        else:
            count_filter_an = None
    elif id == 'count_filter_an':
        if count_filter_an == 1:
            count_filter = allowable_count
        elif count_filter_an == 0:
            count_filter = []
    return count_filter, count_filter_an

# endregion -----------------------------------------------------------------------------------------------------------


# region Data Set Up --------------------------------------------------------------------------------------------------
# Get the Raw Results Data
if exists('assets/all_data.csv'):
    full_df = pd.read_csv('assets/all_data.csv', sep=',', index_col=0)
else:
    full_df = get_all_raw_data()
    full_df.to_csv('assets/all_data.csv')

if DEBUG:
    pass
    # print(full_df)
# endregion -----------------------------------------------------------------------------------------------------------


# region Layout Elements ----------------------------------------------------------------------------------------------
color_mode_switch =  html.Span(
    [
        dbc.Label(className='bi bi-moon', html_for="switch"), # fa fa-moon
        dbc.Switch( id="switch", value=True, className="d-inline-block ms-1", persistence=True,),
        dbc.Label(className='bi bi-brightness-high', html_for="switch"), #fa fa-sun
    ]
)

theme_controls = ThemeChangerAIO(aio_id='theme', 
                                 offcanvas_props={'placement':'end'}, 
                                 button_props={'color':'secondary', 'children':html.I(className='bi bi-gear')},
                                 radio_props={'value':dbc.themes.FLATLY}
                                 )
theme_selection_canvas = theme_controls.children[1]
theme_selection_canvas.children = [color_mode_switch, html.Hr()] + theme_selection_canvas.children


years_ckb = dbc.Checklist(
            options=[dict(zip(['label', 'value'], [name, name])) for name in reversed(years)],
            value=[str(year)],
            id='year_filter',
            style={'padding-left':10},
        )

years_radio = dbc.RadioItems(
    options=[
        {'label':'Last Yr.', 'value':1},
        {'label':'Last 5Yrs.', 'value':5, 'disabled': (len(years) <= 5)},
        {'label':'Last 10Yrs.', 'value':10, 'disabled': (len(years) <= 10)},
        {'label':'All', 'value':0}
    ],
    id='year-multi-select',
    # inline=True,
    style={'padding-left':10},
    value=1
)

year_filter = dbc.DropdownMenu(
            id='yr-dropdown',
            label='Year',
            children=[
                years_radio, # dbc.DropdownMenuItem(, id='multi-year', toggle=False),
                dbc.DropdownMenuItem(divider=True),
                years_ckb, # dbc.DropdownMenuItem(, id='single-year', toggle=False, active=False)
                ],
        )

boat_class_filter = dbc.DropdownMenu(
    label='Boat Class',
    children=[
        dbc.RadioItems(options=[{'label':'All', 'value':1}, {'label':'None', 'value':0}], value=1, style={'padding-left':10}, id='class_filter_an'),
        dbc.DropdownMenuItem(divider=True),
        dbc.Checklist(options=[dict(zip(['label', 'value'], [cl, cl])) for cl in CLASS_LIST], value=CLASS_LIST,
            style={'padding-left':10},
            id='class_filter'
        )
    ],
    # style={'padding':10}
)
overall_position_filter = dbc.DropdownMenu(
    label='Overall Position',
    children=[
        dbc.RadioItems(
            options=[
                {'label':'All', 'value':0},
                {'label':'Top 5', 'value':5},
                {'label':'Top 10', 'value':10},
                {'label':'Top 15', 'value':15},
                {'label':'Top 20', 'value':20},
                {'label':'First Quartile', 'value':25, 'disabled':True},
                {'label':'Second Quartile', 'value':50, 'disabled':True},
                {'label':'Third Quartile', 'value':75,'disabled':True},
            ],
            style={'padding-left':10},
            value=0,
            id='pos_filter'
        )
    ]
)

class_position_filter = dbc.DropdownMenu(
    label='Class Position',
    children=[
        dbc.RadioItems(
            options=[
                {'label':'All', 'value':0},
                {'label':'Top 5', 'value':5},
                {'label':'Top 10', 'value':10},
                {'label':'Top 15', 'value':15},
                {'label':'Top 20', 'value':20},
                {'label':'First Quartile', 'value':25, 'disabled':True},
                {'label':'Second Quartile', 'value':50, 'disabled':True},
                {'label':'Third Quartile', 'value':75,'disabled':True},
            ],
            style={'padding-left':10},
            value=0,
            id='cl_pos_filter'
        )
    ]
)
min_hr = round(full_df['Finish time'].min(), 2) # round(29 + 46/60, 1)
finis_time_filter = html.Div(
    [
        dbc.Label('Finish Time [Hr]'),
        dcc.RangeSlider(min=min_hr, max=100.0, 
                        value=[min_hr, 100],
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='time_filter'
                        ),
    ],
    className="mb-4"
)

gender_filter = dbc.DropdownMenu([
    dbc.RadioItems(options=[{'label':'All', 'value':1}, {'label':'None', 'value':0}], value=1, style={'padding-left':10}, id='gender_filter_an'),
    dbc.DropdownMenuItem(divider=True),
    dbc.Checklist(
        options=[
            {'label':'Undefined', 'value':'Undefined'},
            {'label':'Male', 'value':'Male'},
            {'label':'Female', 'value':'Female'},
            {'label':'Mixed', 'value':'Mixed'},
        ],
        value=['Undefined', 'Male', 'Female', 'Mixed'],
        style={'padding-left':10},
        id='gender_filter'
    )],
    label='Gender'
)

count_filter = dbc.DropdownMenu([
    dbc.RadioItems(options=[{'label':'All', 'value':1}, {'label':'None', 'value':0}], value=1, style={'padding-left':10}, id='count_filter_an'),
    dbc.DropdownMenuItem(divider=True),
    dbc.Checklist(
        options=[
            {'label':'1', 'value':1},
            {'label':'2', 'value':2},
            {'label':'3', 'value':3},
            {'label':'4', 'value':4},
            {'label':'5', 'value':5},
            {'label':'6', 'value':6},
        ],
        value=[1,2,3,4,5,6],
        style={'padding-left':10},
        inline=True,
        id='count_filter'
    )],
    label='Competitor Count'
)

restriction_filter = html.Div(
    [
        dbc.Label('Restrictions',),
        dbc.Switch(label='Rudderless', id='rudder_filter', value=False),
        dbc.Switch(label='Single Blade', id='blade_filter', value=False),
    ],
)
recognition_filter = html.Div(
    [
        dbc.Label('Recognition',),
        dbc.Switch(label='Masters', id='masters_filter', value=False),
        dbc.Switch(label='Adult/Youth', id='adult_youth_filter', value=False),
    ],
)

dropdown2 = html.Div(
    [
        dbc.Label('Data Display Options', className='lg'),
        dcc.Dropdown(
            options=['Time of day', 'Total time', 'Split time', 'Speed'],
            value='Split time',
            id='disp_typ',
            clearable=False,
            style= {'min-width':100}
        ),
    ],
    className='mb-4',
    style= {'width':'auto'}
)

graph_controls = dbc.Stack(
    [
        html.H6('Group By:'),
        dcc.Dropdown(
            options=['None', 'Year', 'Class'],
            value='Year',
            id='group_by',
            clearable=False,
            style= {'min-width':100}
        ),
        dbc.Switch(label='Overlay', id='is_overlay', value=True),
    ],
    direction='horizontal',
    gap=2,
    style={'padding':10}
)

expand_filter_button = dbc.Button([
    html.I(className='bi bi-funnel'),
        '  Data Filters'
],
    id="expand-filter-button",
    className="mb-3",
    color="primary",
    n_clicks=0,
)
        
# This is the card that contains the controls
controls = dbc.Collapse([
    dbc.Card(
        [
        dbc.Label('Data Filters'), # 'Data Filters', 
        dbc.Container([
            dbc.Row(children=dbc.Stack(
                children=[year_filter, boat_class_filter, overall_position_filter, class_position_filter, gender_filter, count_filter],
                direction='horizontal',
                gap=3
                ),
            justify='start'
            ),
            dbc.Row(children=dbc.Stack(
                children=[dropdown2, restriction_filter, recognition_filter, ],
                direction='horizontal',
                gap=5
            ),
            align='start'
            ),
            dbc.Row(children=finis_time_filter),
        ]) 
        ],
        body=True,
    )

    ],
    id="filter-collapse",
    is_open=False,
)

data = filter_data()
grid = dag.AgGrid(
    id="grid",
    defaultColDef={"flex": 1, "minWidth": 40, "sortable": True, "resizable": True,},
    dashGridOptions={'rowSelection':'multiple', 'suppressRowClickSelection':True, 'suppressCellFocus':True},
    style={'--ag-grid-size': 3,
           '--ag-row-height':3},
    columnSize = 'sizeToFit'
)
expand_table_button = dbc.Button([
            html.I(className="bi bi-table"),
            '  Tabular Data'
        ],
            id="collapse-button",
            className="mb-3",
            color="primary",
            n_clicks=0,
        )

collapse = dbc.Collapse(
    dbc.Card(
        [
            dbc.Label('Tabular Data'),
            grid
        ],
        body=True
    ),
    id='collapse',
    is_open=False
)


# region Tabs
animation_tab = dbc.Tab([], 
               label='Animation', 
               class_name='h-4',
               disabled = True
               )
split_tab = dbc.Tab([
    graph_controls,
    dcc.Graph(figure=go.Figure(),
              id='split_graph',
              config = {
                'autosizable':True, 
                'scrollZoom':True,
                'displaylogo':False,
                'displayModeBar': True,
                'modeBarButtonsToRemove':['lasso','select2d', 'resetScale2d']
                },
              style={'height':'auto'}
              )
],
               label='Split Distributions', 
               class_name='h-4',
               style={'height':'auto'}
               )
tab3 = dbc.Tab([], 
               label='River Flow Data', 
               class_name='h-4',
               disabled = True
               )
tab4 = dbc.Tab([], 
               label='Normalized Split Times', 
               class_name='h-4',
               disabled = True
               )

tabs = dbc.Tabs([split_tab,], style={'height':'auto'}) # dbc.Card(dbc.Tabs([split_tab, tab3, tab4, animation_tab]))
# endregion
# endregion -----------------------------------------------------------------------------------------------------------


# region Main Method --------------------------------------------------------------------------------------------------
def main():
    print('Starting the main method------------------------------------------------------------------------------------')
    global DEBUG
    # region Build the app layout

    app.layout = dbc.Container([
        html.Div(dbc.Stack(
            [
                # html.Div('test'),
                html.Div('Texas Water Safari Results', className='mx-auto text-primary text-center fs-3'),
                html.Div(theme_controls)
            ],
            direction='horizontal'
        ), style={'padding':2}),
        dbc.Stack(
            [expand_filter_button, expand_table_button],
            direction='horizontal',
            gap=3,
        ),
        dbc.Row([controls], style={'padding':2}),
        dbc.Row([collapse],style={'padding':2}),
        dbc.Row([tabs], style={'padding':3, 'flex':'auto'},align='stretch'),

        # dcc.Store stores the data value
        dcc.Store(id='data', data=data.to_json(orient='split')),
        dcc.Store(id='selected_teams',)
    ],
    fluid=True,
    class_name="dbc dbc-ag-grid",
    )
    # endregion
    
    app.run(debug=DEBUG)
    print('Successfully reached the end of main -----------------------------------------------------------------------')

# region Callbacks ----------------------------------------------------------------------------------------------------
# updates the Bootstrap global light/dark color mode
clientside_callback(
    """
    switchOn => {       
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');  
       return window.dash_clientside.no_update
    }
    """,
    Output("switch", "id"),
    Input("switch", "value"),
)

# collapsing the data table
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# collapsing the data filters
@app.callback(
    Output('filter-collapse', 'is_open'),
    Input('expand-filter-button', 'n_clicks'),
    State('filter-collapse', 'is_open')
)
def toggle_filter_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# Data Filtered
@app.callback(
    # Outputs
    Output('year-multi-select', 'value'),
    Output('year_filter', 'value'),
    Output('class_filter', 'value'),
    Output('class_filter_an', 'value'),
    Output('gender_filter', 'value'),
    Output('gender_filter_an', 'value'),
    Output('count_filter', 'value'),
    Output('count_filter_an', 'value'),
    Output('data', 'data'),

    # Inputs
    Input('year_filter', 'value'),
    Input('year-multi-select', 'value'),
    Input('class_filter', 'value'),
    Input('class_filter_an', 'value'),
    Input('pos_filter', 'value'),
    Input('cl_pos_filter', 'value'),
    Input('gender_filter', 'value'),
    Input('gender_filter_an', 'value'),
    Input('count_filter', 'value'),
    Input('count_filter_an', 'value'),
    Input('rudder_filter', 'value'),
    Input('blade_filter', 'value'),
    Input('masters_filter', 'value'),
    Input('adult_youth_filter', 'value'),
    Input('time_filter', 'value'),
)
def filter_update(year_filter, multi_value, class_filter, class_filter_an, pos_filter, cl_pos_filter, 
                  gender_filter, gender_filter_an, count_filter, count_filter_an, rudder_filter, blade_filter, 
                  masters_filter, adult_youth_filter, time_filter
                  ):
    # Find the ID of the input that triggered the callback
    trigger_id = ctx.triggered_id

    # Synchronize the input 
    if trigger_id == 'year_filter' or trigger_id == 'year-multi-select':
        multi_value, year_filter = update_year(id=trigger_id, selected_yrs=year_filter, multi_value=multi_value)
    elif trigger_id == 'class_filter' or trigger_id == 'class_filter_an':
        class_filter, class_filter_an = update_class(id=trigger_id, class_filter=class_filter, class_filter_an=class_filter_an)
    elif trigger_id == 'gender_filter' or trigger_id == 'gender_filter_an':
        gender_filter, gender_filter_an = update_gender(id=trigger_id, gender_filter=gender_filter, gender_filter_an=gender_filter_an)
    elif trigger_id == 'count_filter' or trigger_id == 'count_filter_an':
        count_filter, count_filter_an = update_count(id=trigger_id, count_filter=count_filter, count_filter_an=count_filter_an)
    

    # Filter the data
    fd_df = filter_data(year_filter=year_filter, 
                              class_filter=class_filter, pos_filter=pos_filter, cl_pos_filter=cl_pos_filter, 
                              gender_filter=gender_filter, count_filter=count_filter, rudder_filter=rudder_filter, 
                              blade_filter=blade_filter, masters_filter=masters_filter, adult_youth_filter=adult_youth_filter, 
                              time_filter=time_filter
                              )

    # Find teams that are first overall
    # print(fd_df.loc[fd_df['Overall Place'] == '1'])
    # Select the rows of all first place overall teams

    return multi_value, year_filter, class_filter, class_filter_an, gender_filter, gender_filter_an, count_filter, count_filter_an, fd_df.to_json(orient='split')



@app.callback(
    Output('split_graph', 'figure'),
    Input(ThemeChangerAIO.ids.radio('theme'), 'value'),
    Input('switch', 'value'),
    Input('data', 'data'),
    Input('disp_typ', 'value'),
    Input('selected_teams', 'data'),
    Input('group_by', 'value'),
    Input('is_overlay', 'value'),
    State('split_graph', 'figure'),
)
def update_split_graph(theme, switch_on, data, disp_typ, selected_teams, group_by, is_overlay, fig):
  
    gpdict = {
        'Year':'year',
        'None':None,
        'Class':'Class'
    }
    
    if is_overlay:
        violinmode = 'overlay'
    else:
        violinmode = 'group'
    # Find the ID of the input that triggered the callback
    trigger_id = ctx.triggered_id

    # Find the right color template
    template_name = theme.split('/')[-2]
    template = pio.templates[template_name] if switch_on else pio.templates[f'{template_name}_dark']


    # Add data to chart    
    figure_data = pd.read_json(data,orient='split')
    hovertemplate='<b>Boat# %{customdata[0]}</b><br><b>%{customdata[1]}</b><br><br>%{customdata[2]}-Overall, %{customdata[3]}-%{customdata[4]}<br>' + disp_typ + ' for %{x}: %{y}<extra>%{customdata[5]}</extra>'
    hovertemplate_trace=disp_typ + ' for %{x}: %{y}'
    if disp_typ == 'Time of day':
        figure_data[DISP_TYP_DICT[disp_typ]] = pd.to_datetime(figure_data['datetime_2000_1_1'])
        tickformat = '%a %H:%M'
    elif disp_typ == 'Split time':
        figure_data[DISP_TYP_DICT[disp_typ]] = pd.to_timedelta(figure_data[DISP_TYP_DICT[disp_typ]]) + pd.to_datetime('1970/01/01')
        tickformat = '%H:%M'
    elif disp_typ == 'Total time':
        figure_data[DISP_TYP_DICT[disp_typ]] = pd.to_numeric(figure_data['float_hours']) # pd.to_timedelta(figure_data['Hours']) / 3600000000000 # pd.to_datetime(figure_data['datetime_2000_1_1']) - pd.to_datetime('2000/1/1 9:00am') # figure_data['Hours']
        tickformat = '%2f'
        hovertemplate='<b>Boat# %{customdata[0]}</b><br><b>%{customdata[1]}</b><br><br>%{customdata[2]}-Overall, %{customdata[3]}-%{customdata[4]}<br>' + disp_typ + ' for %{x}: %{y:.2f}:Hr.<extra>%{customdata[5]}</extra>'
        hovertemplate_trace=disp_typ + ' for %{x}: %{y:.2f}Hr.'
    elif disp_typ == 'Speed':
        tickformat ='%2f'
        hovertemplate='<b>Boat# %{customdata[0]}</b><br><b>%{customdata[1]}</b><br><br>%{customdata[2]}-Overall, %{customdata[3]}-%{customdata[4]}<br>' + disp_typ + ' for %{x}: %{y:.2f}MPH<extra>%{customdata[5]}</extra>'
        hovertemplate_trace=disp_typ + ' for %{x}: %{y:.2f}MPH'

    fig = px.violin(figure_data, y=DISP_TYP_DICT[disp_typ], x='Split Name', 
    color=gpdict[group_by], 
    points='all', 
    custom_data=['Boat #', 'Team Name', 'Overall Place', 'Class Place', 'Class', 'year'],
    category_orders= {'Split Name':[split for split in TWS_CHECKPOINTS.index if split in figure_data['Split Name'].unique()]},
    template=template
    )

    fig.update_yaxes(tickformat=tickformat, title_text=disp_typ)
    fig.update_traces(
        meanline_visible=True, 
        hovertemplate = hovertemplate,
        pointpos=0,
        hoveron = 'points+kde', #'violins+points+kde'
        opacity=1,
        fillcolor='rgba(225,225,225,0.0)' # Makes the violin fill transparent
    )


    # Add Scatter plots if teams are selected
    if selected_teams != '{"columns":[],"index":[],"data":[]}':
        # Get the teams that need to pe 
        teams = pd.read_json(selected_teams,orient='split')[['year', 'Overall Place', 'Class']]

        
        # Are there more then one violin plots
        if len(fig.data) > 1:
            # In this case we need to figure out the color that each each trace should be
            colors = {}
            line_is = {}
            # dashes 
            for violin in fig.data:
                color = violin.marker.color
                key = str(violin.name)
                colors.update({key:color})
                line_is.update({key:0})
            # line_i = 0
            line_dashs = {0:'solid', 1:'dot', 2:'dash', 3:'longdash', 4:'dashdot', 5:'longdashdot'}
            for team in teams.iterrows():
                team = team[1]
                color_key = str(team['year']) if group_by == 'Year' else str(team['Class'])
                name = figure_data.loc[(figure_data['year'] == team['year']) & (figure_data['Overall Place'] == team['Overall Place']), 'Team Name'].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=figure_data.loc[(figure_data['year'] == team['year']) & (figure_data['Overall Place'] == team['Overall Place']), 'Split Name'],
                        y=figure_data.loc[(figure_data['year'] == team['year']) & (figure_data['Overall Place'] == team['Overall Place']), DISP_TYP_DICT[disp_typ]],
                        mode='lines',
                        name=name,
                        line={'color':colors[color_key], 'dash':line_dashs[line_is[color_key]]},
                        # customdata=[str(team['year'])],
                        hovertemplate=f'<b>{name}</b><br>' + hovertemplate_trace + f'<extra>{team.year}</extra>',
                    )
                )
            
                line_is[color_key] = 0 if line_is[color_key] == 5 else line_is[color_key] + 1
        else:
            lines_data = figure_data.loc[(figure_data['year'].isin(teams['year'])) & (figure_data['Overall Place'].isin(teams['Overall Place']))]
            lines = px.line(lines_data, x='Split Name', y=DISP_TYP_DICT[disp_typ], template=template, color='Team Name',
                custom_data=['Boat #', 'Team Name', 'Overall Place', 'Class Place', 'Class', 'year']
            )
            lines.update_traces(
                    hovertemplate= hovertemplate
            )
            for line in lines.data:
                fig.add_trace(line)

    fig.update_layout(height=600, violinmode=violinmode,)
    return fig


@app.callback(
    Output('grid', 'columnDefs'),
    Output('grid', 'rowData'),
    Input('data', 'data'),
    Input('disp_typ', 'value'),
)
def update_table(data, disp_typ,):
    # Read data into DataFrame
    df = pd.read_json(data,orient='split')

    # Pivot DataFrame to make a row for each team
    df = df.pivot(columns='Split Name', values=DISP_TYP_DICT[disp_typ], index=['year', 'Overall Place', 'Class Place', 'Class', 'Team Members'])
    
    # Sort the columns
    df = df.reindex([cp for cp in TWS_CHECKPOINTS.index if cp in df.columns], axis=1)

    # Make the index cols into data cols
    df.reset_index(inplace=True)

    # Create dict of column definitions 
    columnDefs = [{'field': f,
                 'filter':(i==4),
                 'wrapText':(i==4),
                 'sortable':(i!=4),
                 'autoHeight': True,
                 'minWidth': 80 + (i==4)*360 - 40*((i==0)|(i==1)|(i==2)|(i==3)),
                 'checkboxSelection':(i==4),
                 'headerCheckboxSelection':(i==4),
                 'headerCheckboxSelectionFilteredOnly':True
                 } for i, f in enumerate(df.columns)]
    return columnDefs, df.to_dict('records')


@app.callback(
    Output('selected_teams', 'data'),
    Input('grid', 'selectedRows')
)
def team_selected(rows):
    df = pd.DataFrame(rows)
    return df.to_json(orient='split')
# endregion -----------------------------------------------------------------------------------------------------------


# Call the main method
# if __name__ == '__main__':
#     main()

main()

'''
TODO
    *
'''