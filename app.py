# region Imports ------------------------------------------------------------------------------------------------------
from dash import Dash, html, dash_table, dcc, Input, Output, State, callback, Patch, clientside_callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

import pandas as pd

import gpxpy
import gpxpy.gpx

from math import radians, cos, sin, asin, sqrt
# endregion imports ---------------------------------------------------------------------------------------------------


# region Globals  -----------------------------------------------------------------------------------------------------
# Global Constants
TWS_ROUTE = gpxpy.parse(open('data\\tws_race_route.gpx', 'r'))
TWS_TOTAL_MILES: float
TWS_CHECKPOINTS: pd.DataFrame

# Global Variables
# stylesheet with the .dbc class to style  dcc, DataTable and AG Grid components with a Bootstrap theme
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

external_stylesheets = [dbc.themes.SOLAR, dbc.icons.FONT_AWESOME, dbc_css]
app = Dash(external_stylesheets=external_stylesheets)
year = 2024
data: pd.DataFrame
# endregion globals ---------------------------------------------------------------------------------------------------


# region helper methods -----------------------------------------------------------------------------------------------
def get_raw_data(year: int) -> pd.DataFrame:
    print('Enter: get_raw_data ---------------\n')
    # This method reads the CSV split spreadsheet data as downloaded from https://www.texaswatersafari.org/

    # Read the data CSV file
    file_str = 'Data/' + str(year) + '/' + str(year) + '.csv'
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


    # Add columns containing the boat class and special designation (if any) for each team
    df['Recognition'] = df['Recognition'].str.split(' ', n=1).str[1]
    df['Class'] = df['Recognition'].str.split('\n|\r', n=1, regex=True).str[0]

    df['Class'] = df['Class'].str.replace(r"^ +| +$", r'', regex=True) #.strip()
    df['Class'] = df['Class'].str.replace('C-2','USCA C-2', regex=False) #df.loc[df['Class'] == 'C-2', 'Class'] = 'USCA C-2'
    df['Class'] = df['Class'].str.replace('C-1 Man','USCA C-1 Man', regex=False) # df.loc[df['Class'] == 'C-1 Man', 'Class'] = 'USCA C-1 Man'
    df['Class'] = df['Class'].str.replace('USCA USCA','USCA', regex=False)
    df['Class'] = df['Class'].str.replace('Unlimited Man','Solo Unlimited Man', regex=False) # df.loc[df['Class'] == 'Unlimited Man', 'Class'] = 'Solo Unlimited Man'
    df['Class'] = df['Class'].str.replace('Solo Solo','Solo', regex=False)
    

    #Split out competitors and team captions
    df[['Competitors', 'Team Captions']] = pd.DataFrame(df['Team Members'].str.split(pat='TC ', n=1 , regex=True).to_list(), index=df.index)

    # Convert str to list of str
    df['Competitors'] = df['Competitors'].str.replace('\n{2,}', '').str.split('\r\n', regex=True)
    df['Team Captions'] = df['Team Captions'].str.replace('TC |\n{2,}', '').str.split('\r\n',regex=True)

    # Convert Boat Number to Int
    df['Boat #'] = df['Boat #'].astype(int)

    print('Exit: get_raw_data ---------------\n')
    return df

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
# endregion -----------------------------------------------------------------------------------------------------------


# region Data Set Up --------------------------------------------------------------------------------------------------
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
print(TWS_CHECKPOINTS)

# Get the Raw Results Data
data = get_raw_data(year=year)

# Format newlines for plotly to recognize
data['Team Members'] = data['Team Members'].str.replace('\n\n', '\n')
data['Team Members'] = data['Team Members'].str.replace('\n', '; ')


df = data.loc[:,'Overall Place':'Seadrift']
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

dropdown = html.Div(
    [
        dbc.Label("Select indicator (y-axis)"),
        dcc.Dropdown(
            ["element1", "element2", "element3"],
            "pop",
            id="indicator",
            clearable=False,
        ),
    ],
    className='mb-4',
)

# This is the card that contains the controls
controls = dbc.Card(
    [dropdown],
    body=True
)
grid = dag.AgGrid(
    id="grid",
    columnDefs=[{"field": f,
                 'filter':(i<=3),
                 'wrapText':(i==2),
                 "autoHeight": True
                 } for i, f in enumerate(df.columns)],
    rowData=df.to_dict("records"),
    defaultColDef={"flex": 1, "minWidth": 50, "sortable": False, "resizable": True},
    dashGridOptions={"rowSelection":"multiple"},
    style={'--ag-grid-size': 3,
           '--ag-row-height':3},
)
table = dash_table.DataTable(
    id='table',
    data=df.to_dict('records'),
    style_header={

    }
)
# region Tabs
tab1 = dbc.Tab([grid], 
               label='Tabular Splits', 
               class_name='h-4'
               )
tabs = dbc.Card(dbc.Tabs([tab1]))
# endregion
# endregion -----------------------------------------------------------------------------------------------------------


# region Main Method --------------------------------------------------------------------------------------------------
def main():
    print('Starting the main method------------------------------------------------------------------------------------')

    # region Build the app layout
    app.layout = dbc.Container([
        dbc.Row([
            theme_controls,
            html.Div('Texas Water Safari Results', className='text-primary text-center fs-3')
        ]),
        dbc.Row([controls], style={'padding':3}),
        dbc.Row([tabs], style={'padding':3})

    ],
    fluid=True,
    class_name="dbc dbc-ag-grid",
    )
    
    """     [
        html.Div(children='Tabular Split Data'),
        dash_table.DataTable(
            id='table',
            data=df.to_dict('records'),
            style_header={
                # 'backgroundColor': 'rgb(30, 30, 30)',
                # 'color': 'white',
                # 'font-family':'Arial',
                # 'font-weight':'bold',
                'overflow':'hidden',
                'overflow-wrap':'break-word',
                'maxWidth':0,
                'height':'auto',
                },
            style_data={
                # 'backgroundColor': 'rgb(50, 50, 50)',
                # 'color': 'white',
                # 'font-family':'Arial',
                'whiteSpace': 'per-line',
                'width':'auto',
                'overflow-wrap':'normal',
                'height':'auto',
                'lineHeight': '3px'
            },
            columns=[{'id': c, 'name': c, 'presentation':'markdown'} for c in df.columns],
            # # page_size=20,
            style_cell={
                'overflow-x':'hidden',
                'overflow-y':'visible',
                'textOverflow':'ellipsis',
                # 'maxWidth':0,
                'textAlign':'left'
            }
            )
            
        ] """
    # endregion
    
    app.run(debug=True)
    print('Successfully reached the end of main -----------------------------------------------------------------------')

# region Callbacks ----------------------------------------------------------------------------------------------------
@callback(
    Output("line-chart", "figure" ),
    Output("scatter-chart", "figure"),
    Output("grid", "dashGridOptions"),
    Input("indicator", "value"),
    Input("continents", "value"),
    Input("years", "value"),
    State(ThemeChangerAIO.ids.radio("theme"), "value"),
    State("switch", "value"),
)
def update(indicator, continent, yrs, theme, color_mode_switch_on):

    if continent == [] or indicator is None:
        return {}, {}, {}

    theme_name = template_from_url(theme)
    template_name = theme_name if color_mode_switch_on else theme_name + "_dark"

    dff = df[df.year.between(yrs[0], yrs[1])]
    dff = dff[dff.continent.isin(continent)]

    fig = px.line(
        dff,
        x="year",
        y=indicator,
        color="continent",
        line_group="country",
        template=template_name
    )

    fig_scatter = px.scatter(
        dff[dff.year == yrs[0]],
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        log_x=True,
        size_max=60,
        template=template_name,
        title="Gapminder %s: %s theme" % (yrs[1], template_name),
    )


    grid_filter = f"{continent}.includes(params.data.continent) && params.data.year >= {yrs[0]} && params.data.year <= {yrs[1]}"
    dashGridOptions = {
        "isExternalFilterPresent": {"function": "true"},
        "doesExternalFilterPass": {"function": grid_filter},
    }

    return fig, fig_scatter, dashGridOptions


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


# This callback makes updating figures with the new theme much faster
# @callback(
#     Output("line-chart", "figure", allow_duplicate=True ),
#     Output("scatter-chart", "figure", allow_duplicate=True),
#     Input(ThemeChangerAIO.ids.radio("theme"), "value"),
#     Input("switch", "value"),
#     prevent_initial_call=True,
# )
# def update_template(theme, color_mode_switch_on):
#     theme_name = template_from_url(theme)
#     template_name = theme_name if color_mode_switch_on else theme_name + "_dark"

#     patched_figure = Patch()
#     # When using Patch() to update the figure template, you must use the figure template dict
#     # from plotly.io  and not just the template name
#     patched_figure["layout"]["template"] = pio.templates[template_name]
#     return patched_figure, patched_figure

# # updates the Bootstrap global light/dark color mode
# clientside_callback(
#     """
#     switchOn => {       
#        document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');  
#        return window.dash_clientside.no_update
#     }
#     """,
#     Output("switch", "id"),
#     Input("switch", "value"),
# )

# endregion -----------------------------------------------------------------------------------------------------------


# Call the main method
if __name__ == '__main__':
    main()