# region Imports ------------------------------------------------------------------------------------------------------
import pandas as pd
import app
# from app import find_class, find_gender, find_hull_ln, find_hull_width, is_rudder, is_double_blade, is_masters, is_adult_youth, TWS_CHECKPOINTS, name_formatter, cont_competitors

# region vars ---------------------------------------------------------------------------------------------------------
year = 2009
# endregion -----------------------------------------------------------------------------------------------------------

# endregion -----------------------------------------------------------------------------------------------------------
def get_raw_data(year: int) -> pd.DataFrame:
    # print('Enter: get_raw_data ---------------\n')
    # This method reads the CSV split spreadsheet data as downloaded from https://www.texaswatersafari.org/

    # Read the data CSV file
    file_str = 'assets/split_data/' + str(year) + '/' + str(year) + '.csv'
    df = pd.read_csv(file_str, sep=',', header=6)

    # Get the names on each team member into one cell
    df['Team Members'] = df['Team Members'].astype(str)
    # team_names = []
    # team_name = ''
    # new_flag = True
    # for i, name in df['Team Members'].items():
    #     if new_flag and name != 'nan':
    #         team_name = name # Star new team name
    #         new_flag = False
    #     elif name != 'nan':
    #         # Add name to team_name
    #         team_name += ' \r; ' + name
    #     else:
    #         # Add team_name to team_names and set new_flag to true
    #         team_names.append(team_name)
    #         new_flag = True
    # team_members = pd.Series(team_names).drop_duplicates(ignore_index=True)

    # Rename some columns. This is needed because the headers titles in the CSV do not align with the values (Why TWS?!?!).
    # Start with the colum named 'Staples' and iterate through the data columns
    for n in range(df.columns.get_loc('Staples'), len(df.columns)-1, 2): 
        df = df.rename(columns={df.columns[n]:'Unnamed', df.columns[n+1]:df.columns[n]})

    df = df.rename(columns={'Special Recognition':'Recognition', 'Position':'Overall Place'})


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

    # df['Team Members'] = team_members

    # Add columns containing the boat class
    df['Class'] = df['Recognition'].apply(app.find_class)
    df['Gender'] = df['Class'].apply(app.find_gender)
    df['Max Boat Len'] = df['Class'].apply(app.find_hull_ln)
    df['Min Boat Width'] = df['Class'].apply(app.find_hull_width)
    df['Rudder'] = df['Class'].apply(app.is_rudder)
    df['Double Blade'] = df['Class'].apply(app.is_double_blade)

    # Add column for any special recognitions
    df['Masters'] = df['Recognition'].apply(app.is_masters)
    df['Adult Youth'] = df['Recognition'].apply(app.is_adult_youth)
    

    #Split out competitors and team captions
    df[['Competitors', 'Team Captions']] = pd.DataFrame(df['Team Members'].astype(str).str.split(pat='TC ', n=1 , regex=True).to_list(), index=df.index)

    # Convert str to list of str
    df['Competitors'] = df['Competitors'].str.replace('\n{2,}', '').str.split('\r\n', regex=True)
    df['Competitors'] = df['Competitors'].apply(lambda a: list(filter(None, list(a))))
    df['Team Captions'] = df['Team Captions'].str.replace('TC |\n{2,}', '').str.split('\r\n',regex=True)

    # Add column for competitor count
    df['Competitor count'] = df['Competitors'].apply(lambda x: len(x))

    # Convert Boat Number to Int
    print(df['Boat #'])
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
    df['Milage'] = df['Split Name'].apply(lambda a: app.TWS_CHECKPOINTS.loc[a, 'Milage'])

    # Add a column with the split milage
    this_year_cp_list = app.TWS_CHECKPOINTS.loc[df['Split Name'].unique()]
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
    df['Team Name'] = df['Team Members'].apply(app.name_formatter)

    # Get compeditor count
    df['Competitor count'] = df['Team Members'].apply(app.cont_competitors)

    # print('Exit: get_raw_data ---------------\n')
    return df

def main():
    data = get_raw_data(year)
    print(data)
    existing = pd.read_csv('assets/all_data.csv')
    print(existing)
    data = pd.concat([existing, data], ignore_index=True)
    print(data)
    print(round(data['Finish time'].min(), 2))
    data.to_csv(f'assets/split_data/{year}/{year}_cleaned.csv')

if __name__ == '__main__':
    main()