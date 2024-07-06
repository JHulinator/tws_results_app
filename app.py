# region Imports ------------------------------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore') # , category=FutureWarning
from dash import Dash, html, dash_table, dcc, Input, Output, State, callback, Patch, clientside_callback, ctx, set_props
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

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
# endregion imports ---------------------------------------------------------------------------------------------------


# region Globals  -----------------------------------------------------------------------------------------------------
# Global Constants
TWS_ROUTE = gpxpy.parse(open('data\\tws_race_route.gpx', 'r'))
TWS_TOTAL_MILES: float
TWS_CHECKPOINTS: pd.DataFrame
DEBUG = True
CLASS_LIST = list(pd.read_csv('data\\class_list.csv', sep=',')['class'])

# Global Variables
# stylesheet with the .dbc class to style  dcc, DataTable and AG Grid components with a Bootstrap theme
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

external_stylesheets = [dbc.themes.SOLAR, dbc.icons.FONT_AWESOME, dbc_css]
app = Dash(external_stylesheets=external_stylesheets)
year = 2024
years = os.listdir('data\\split_data\\')
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
TWS_CHECKPOINTS = pd.read_csv('data\\tws_checkpoint_list.csv', sep=',', index_col=0)

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
    str_class = str_class.replace('masters', '').replace('adultyouth', '').replace('adultchild', '')
    

    # find the class that is the most similar to the string
    similarity_ratio = 0
    return_class = None
    for cl in CLASS_LIST:
        if SequenceMatcher(None, str_class, cl.lower()).ratio() > similarity_ratio:
            similarity_ratio = SequenceMatcher(None, str_class, cl.lower()).ratio()
            return_class = cl

    return return_class

def find_gender(str_class:str) -> str:
    df_class = pd.read_csv('data\\class_list.csv', sep=',', index_col='class')['gender']
    return str(df_class.loc[str_class])

def find_hull_ln(str_class:str) -> str:
    df_class = pd.read_csv('data\\class_list.csv', sep=',', index_col='class')['hull lenght restriction']
    return str(df_class.loc[str_class])

def find_hull_width(str_class:str) -> str:
    df_class = pd.read_csv('data\\class_list.csv', sep=',', index_col='class')['hull width restriction']
    return str(df_class.loc[str_class])

def is_rudder(str_class:str) -> bool:
    df_class = pd.read_csv('data\\class_list.csv', sep=',', index_col='class')['rudder restriction']
    return not bool(df_class.loc[str_class])

def is_double_blade(str_class:str) -> bool:
    df_class = pd.read_csv('data\\class_list.csv', sep=',', index_col='class')['double blade restriction']
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

def get_raw_data(year: int) -> pd.DataFrame:
    # print('Enter: get_raw_data ---------------\n')
    # This method reads the CSV split spreadsheet data as downloaded from https://www.texaswatersafari.org/

    # Read the data CSV file
    file_str = 'Data\\split_data\\' + str(year) + '/' + str(year) + '.csv'
    df = pd.read_csv(file_str, sep=',', header=6)

    # Rename some columns. This is needed because the headers titles in the CSV do not align with the values (Why TWS?!?!).
    # Start with the colum named 'Staples' and iterate through the data columns
    for n in range(df.columns.get_loc('Staples'), 30, 2): 
        df = df.rename(columns={df.columns[n]:'Unnamed', df.columns[n+1]:df.columns[n]})

    # The CSV file has a lot of empty cells (for excel hell formatting). We are now going to get rid of these
    # meaningless cells.
    df.drop(df.columns[df.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True) # This Drops all the unnamed columns
    df.iloc[:, 4:] = df.iloc[:, 4:].shift(-2) # This shifts the time data up so that total cumulative time is aligned with team info line. This will be the only data that is kept
    df = df[df[df.columns[0]].notna()] # This drops all the rows that don't have info in the first column
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
    df[['Competitors', 'Team Captions']] = pd.DataFrame(df['Team Members'].str.split(pat='TC ', n=1 , regex=True).to_list(), index=df.index)

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

    # Create columns with formatted split times: time_of_day, total_time, split_time
    cp_cols = list(TWS_CHECKPOINTS.axes[0].values[1:])
    day_dict = {1:'Sat', 2:'Sun', 3:'Mon', 4:'Tue', 5:'Wed'}
    
    last_cp = None
    for i, name in enumerate(cp_cols):
        if name in df.columns:
            df[f'{name}_TOD'] = df[name].apply(lambda x: f'{day_dict.setdefault(1+x.days+1*(divmod(x.seconds, 3600)[0]>15))} {9+divmod(x.seconds, 3600)[0]}:{divmod(divmod(x.seconds, 3600)[1], 60)[0]}')
            df[f'{name}_TT'] = df[name].apply(lambda x: f'{divmod(x.seconds, 3600)[0] + 24*x.days}:{divmod(divmod(x.seconds, 3600)[1], 60)[0]}:{divmod(divmod(x.seconds, 3600)[1], 60)[1]}') #.astype(str)
            if i == 0:
                df[f'{name}_ST'] = df[f'{name}_TT']
                last_cp = name
            else:
                df[f'{name}_ST'] = (df[name] - df[last_cp]).apply(lambda x: f'{divmod(x.seconds, 3600)[0] + 24*x.days}:{divmod(divmod(x.seconds, 3600)[1], 60)[0]}:{divmod(divmod(x.seconds, 3600)[1], 60)[1]}')
                last_cp = name
            m = TWS_CHECKPOINTS.loc[name, 'Milage'] - TWS_CHECKPOINTS['Milage'].iloc[i]
            h = str_to_hours(df[f'{name}_ST']) 
            df[f'{name}_SS'] = m / h
            df[f'{name}_SS'].loc[df[f'{name}_SS'] == np.inf] = 0.0
    

    # Add column for class place
    df['Class Place'] = df['Recognition'].str.split(' ').apply(lambda a: a[0])
    df['Class Place'] = df['Class Place'].str.replace(r'\D', '', regex=True)

    # Add column for finish time
    df['Hrs'] = df['Seadrift'].apply(lambda x: x.total_seconds()/3600)

    
    # print('Exit: get_raw_data ---------------\n')
    return df


def get_all_raw_data() -> pd.DataFrame:
    print('Enter get all raw data ------------------')
    df = pd.DataFrame()
    for yr in years:
        yr_df = get_raw_data(yr)
        yr_df['year'] = yr
        df = pd.concat([df, yr_df], ignore_index=True, sort=False)
    print('Exit get all raw data -------------------')
    return df

# def add_sufix(s, adder):
#     return s + adder

def filter_data(df:pd.DataFrame, disp_typ:str='Time of day', year_filter:List[int]=years, class_filter:List[str]=CLASS_LIST,
                pos_filter:int=0, cl_pos_filter:int=0, gender_filter:List[str]=['Undefined', 'Male', 'Female', 'Mixed'],
                count_filter:List[int]=[1,2,3,4,5,6], rudder_filter:bool=False, blade_filter:bool=False, 
                masters_filter:bool=False, adult_youth_filter:bool=False, time_filter:List[float]=[0, 100]) -> pd.DataFrame:
    
    year_filter = list(map(int, year_filter)) # Convert to int if not already
    if pos_filter == 0: pos_filter = np.inf
    if cl_pos_filter == 0: cl_pos_filter = np.inf

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
                ((df['Hrs'] >= time_filter[0]) & (df['Hrs'] <= time_filter[1]))
                ]
    cps = list(TWS_CHECKPOINTS.axes[0].values[1:])
    # ['Time of day', 'Total time', 'Split time', 'Speed']
    if disp_typ == 'Time of day':
        key = '_TOD'
    elif disp_typ == 'Total time':
        key = '_TT'
    elif disp_typ == 'Split time':
        key = '_ST'
    elif disp_typ == 'Speed':
        key = '_S'
    else:
        key = ''

    data_cols = [x + key for x in cps]
    
    cols = ['year', 'Overall Place', 'Team Members'] + data_cols
    filtered = filtered.loc[:, cols]
    
    filtered.rename(columns=dict(zip(data_cols, cps)), inplace=True)
    filtered.reset_index(inplace=True, drop=True)
    return filtered
# endregion -----------------------------------------------------------------------------------------------------------


# region Data Set Up --------------------------------------------------------------------------------------------------
# Get the Raw Results Data
if exists('all_data.csv'):
    df = pd.read_csv('all_data.csv', sep=',', index_col=0)
else:
    df = get_all_raw_data()
    df.to_csv('all_data.csv')

if DEBUG:
    print(df)
# endregion -----------------------------------------------------------------------------------------------------------


# region Layout Elements ----------------------------------------------------------------------------------------------
color_mode_switch =  html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="switch"),
        dbc.Switch( id="switch", value=True, className="d-inline-block ms-1", persistence=True,),
        dbc.Label(className="fa fa-sun", html_for="switch"),
    ]
)
# The ThemeChangerAIO loads all 52  Bootstrap themed figure templates to plotly.io
theme_controls = html.Div(
    [ThemeChangerAIO(aio_id="theme"), color_mode_switch],
    className="hstack gap-3 mt-2"
)

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
        dbc.RadioItems(options=[{'label':'All', 'value':1}, {'label':'None', 'value':0}], value=1, style={'padding-left':10}),
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
                {'label':'First Quartile', 'value':25},
                {'label':'Second Quartile', 'value':50},
                {'label':'Third Quartile', 'value':75},
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
                {'label':'First Quartile', 'value':25},
                {'label':'Second Quartile', 'value':50},
                {'label':'Third Quartile', 'value':75},
            ],
            style={'padding-left':10},
            value=0,
            id='cl_pos_filter'
        )
    ]
)
min_hr = df['Hrs'].min()
finis_time_filter = html.Div(
    [
        dbc.Label('Finish Time [Hr]'),
        dcc.RangeSlider(min=min_hr, max=100.0, 
                        value=[min_hr, 100],
                        tooltip={"placement": "bottom", "always_visible": True}),
                        #id='f-time'
    ],
    className="mb-4"
)

gender_filter = dbc.DropdownMenu([
    dbc.RadioItems(options=[{'label':'All', 'value':1}, {'label':'None', 'value':0}], value=1, style={'padding-left':10}),
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
    dbc.RadioItems(options=[{'label':'All', 'value':1}, {'label':'None', 'value':0}], value=1, style={'padding-left':10}),
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
        dbc.Label('Data Display Options'),
        dcc.Dropdown(
            options=['Time of day', 'Total time', 'Split time', 'Speed'],
            value='Time of day',
            id='disp_typ',
            clearable=False,
        )
    ],
    className='mb-4',
)

# This is the card that contains the controls
controls = dbc.Card([
    dbc.Label('Data Filters'),
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
# cols = ['Overall Place', 'Team Members'] + list(TWS_CHECKPOINTS.axes[0].values[1:])
# print(filter_data(df=df,disp_typ='Time of day'))
data = filter_data(df=df,disp_typ='Time of day')
grid = dag.AgGrid(
    id="grid",
    columnDefs=[{"field": f,
                 'filter':(i==2),
                 'wrapText':(i==2),
                 "autoHeight": True,
                 'minWidth': 80 + (i==2)*360 - 40*((i==0)|(i==1))
                 } for i, f in enumerate(data.columns)],
    rowData= data.to_dict("records"), # df.loc[:,'Overall Place':'Boat #'].to_dict("records"),
    defaultColDef={"flex": 1, "minWidth": 40, "sortable": False, "resizable": True,},
    dashGridOptions={"rowSelection":"multiple"},
    style={'--ag-grid-size': 3,
           '--ag-row-height':3},
    columnSizeOptions={'skipHeader':True, 'keys':list(data.columns[3:].values)},
    columnSize = 'sizeToFit'
)
collapse = html.Div(
    [
        dbc.Button(
            "Tabular Data",
            id="collapse-button",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse(
            grid,
            id="collapse",
            is_open=False,
        ),
    ]
)


# region Tabs
tab1 = dbc.Tab([], 
               label='Animation', 
               class_name='h-4'
               )
tab2 = dbc.Tab([], 
               label='Split Times', 
               class_name='h-4'
               )
tab3 = dbc.Tab([], 
               label='River Flow Data', 
               class_name='h-4'
               )
tab4 = dbc.Tab([], 
               label='Normalized Split Times', 
               class_name='h-4'
               )

tabs = dbc.Card(dbc.Tabs([tab1, tab2, tab3, tab4]))
# endregion
# endregion -----------------------------------------------------------------------------------------------------------


# region Main Method --------------------------------------------------------------------------------------------------
def main():
    print('Starting the main method------------------------------------------------------------------------------------')
    global DEBUG
    # region Build the app layout

    app.layout = dbc.Container([
        dbc.Row([
            theme_controls,
            html.Div('Texas Water Safari Results', className='text-primary text-center fs-3')
        ]),
        dbc.Row([controls], style={'padding':3}),
        dbc.Row([collapse],style={'padding':3}),
        dbc.Row([tabs], style={'padding':3},align='stretch')
    ],
    fluid=True,
    class_name="dbc dbc-ag-grid",
    )
    # endregion
    
    app.run(debug=DEBUG)
    print('Successfully reached the end of main -----------------------------------------------------------------------')

# region Callbacks ----------------------------------------------------------------------------------------------------
""" @callback(
    #Output("line-chart", "figure" ),
    #Output("scatter-chart", "figure"),
    # Output("grid", "dashGridOptions"),
    #Output('debug-text','children'),
    # Output('year_filter', 'value'),
    Output('grid','rowData'),
    Input('year_filter', 'value'),
    # Input('disp_typ', 'value'),
    #Input("continents", "value"),
    #Input("years", "value"),
    State(ThemeChangerAIO.ids.radio("theme"), "value"),
    State("switch", "value"),
    State('disp_typ', 'value')
)
def update_data_table(year_select,theme, color_mode_switch_on, display_as): #continent, yrs,
    global years, df
    print(f'Callback initiated:\n\
            year_select = {year_select}\n\
            Display as: {display_as}',
            #theam = {theme}\n\
            #color_mode_switch = {color_mode_switch_on}',
            flush=False
        )

    # if year_select is None:
    #     year_select = max(years)
    #     print(f'New year assigned = {year_select}')

    # if continent == [] or year_select is None:
    #     return {}, {}, {}

    theme_name = template_from_url(theme)
    template_name = theme_name if color_mode_switch_on else theme_name + "_dark"

    # update dataFrame
    df = get_raw_data(year_select)

    col_list = list(df.iloc[:, :4].columns)
    if display_as == 'Time of day':
        key = '_TOD'
    elif display_as == 'Total time':
        key = '_TT'
    elif display_as == 'Split time':
        key = '_ST'
    elif display_as == 'Speed': # Speed
        key = '_SS'

    col_list += list(df.loc[:, df.columns.str.contains(key)].columns.values)
    print(col_list)
    return_df = df.loc[:, col_list]
    return_df.columns = return_df.columns.str.replace(key, '',)
    print(return_df)

    # dff = df[df.year.between(yrs[0], yrs[1])]
    # dff = dff[dff.continent.isin(continent)]

    # fig = px.line(
    #     dff,
    #     x="year",
    #     y=year_select,
    #     color="continent",
    #     line_group="country",
    #     template=template_name
    # )

    # fig_scatter = px.scatter(
    #     dff[dff.year == yrs[0]],
    #     x="gdpPercap",
    #     y="lifeExp",
    #     size="pop",
    #     color="continent",
    #     log_x=True,
    #     size_max=60,
    #     template=template_name,
    #     title="Gapminder %s: %s theme" % (yrs[1], template_name),
    # )

    # grid.Data = get_raw_data(year=continent).loc[:,'Overall Place':'Seadrift'].to_dict('records')

    # grid_filter = f"{continent}.includes(params.data.continent) && params.data.year >= {yrs[0]} && params.data.year <= {yrs[1]}"
    # dashGridOptions = {
    #     "isExternalFilterPresent": {"function": "true"},
    #     "doesExternalFilterPass": {"function": grid_filter},
    # }

    return return_df.to_dict("records") #f'Callback initiated:\nyear_select = {year_select}' # fig, fig_scatter, dashGridOptions """


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


# Selecting the year
@app.callback(
    Output('year-multi-select', 'value', allow_duplicate=False),
    Output('year_filter', 'value'),
    Output('grid', 'rowData'),
    Input('year_filter', 'value'),
    Input('year-multi-select', 'value'),
    State('disp_typ', 'value'),
    State('class_filter', 'value'),
    State('pos_filter', 'value'),
    State('cl_pos_filter', 'value'),
    State('gender_filter', 'value'),
    State('count_filter', 'value'),
    State('rudder_filter', 'value'),
    State('blade_filter', 'value'),
    State('masters_filter', 'value'),
    State('adult_youth_filter', 'value'),
    #State('f-', 'value'),# 9
    # prevent_initial_call=True
)
def year_selected(selected_yrs, multi_value, disp_typ, class_filter,pos_filter, cl_pos_filter, gender_filter,count_filter, 
                    rudder_filter, blade_filter, masters_filter, adult_youth_filter):
    

    trigger_id = ctx.triggered_id
    last5years = [years[i] for i in np.argsort(years)[-5:]]
    last10years = [years[i] for i in np.argsort(years)[-10:]]
    if trigger_id == 'year_filter':
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
    elif trigger_id == 'year-multi-select':
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
    year_filter = return_val[1]

    new_data = filter_data(
        df=df, disp_typ=disp_typ, year_filter=year_filter, class_filter=class_filter, rudder_filter=rudder_filter,
        blade_filter=blade_filter, masters_filter=masters_filter, adult_youth_filter=adult_youth_filter, 
    )

    # return_val.append(new_data.to_dict('records'))
    return return_val[0], return_val[1], new_data.to_dict('records')


# Selecting Data display option
@app.callback(
    Output('grid', 'rowData', allow_duplicate=True),
    Input('disp_typ', 'value'),
    State('year_filter', 'value'),
    State('class_filter', 'value'),
    State('pos_filter', 'value'),
    State('cl_pos_filter', 'value'),
    State('gender_filter', 'value'),
    State('count_filter', 'value'),
    State('rudder_filter', 'value'),
    State('blade_filter', 'value'),
    State('masters_filter', 'value'),
    State('adult_youth_filter', 'value'),
    prevent_initial_call=True
)
def disp_typ_selected(disp_typ, year_filter, class_filter,pos_filter, cl_pos_filter, gender_filter,count_filter, 
                    rudder_filter, blade_filter, masters_filter, adult_youth_filter):
    return filter_data(
        df=df, disp_typ=disp_typ, year_filter=year_filter, class_filter=class_filter, rudder_filter=rudder_filter,
        blade_filter=blade_filter, masters_filter=masters_filter, adult_youth_filter=adult_youth_filter,
    ).to_dict('records')

# @app.callback(
#     Output('year_filter', 'value'),
#     Input('year-multi-select', 'value'),
#     State('year_filter', 'value')
# )
# def mlti_year_select(multi_value, selected_yrs):
#     if multi_value == 1:
#         # Ensure that only the first year is selected
#         if not selected_yrs == [max(years)]:
#             # make them equal
#             return [max(years)]
#         else:
#             pass
#     elif multi_value == 5:
#         pass
#     elif multi_value == 10:
#         pass
#     elif multi_value == 0:
#         pass
#     else:
#         pass


# endregion -----------------------------------------------------------------------------------------------------------


# Call the main method
if __name__ == '__main__':
    main()

'''TODO
    * Add callback for data type selection
    * Fix speed calculations
    * Fix C1 getting classified as C2
'''