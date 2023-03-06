# Imports #
import os
import pandas as pd
import time
import resource
import numpy as np
####################

# Parameters #
dataset_path = '../data/states_30_11_2022.csv'
_verbose = True
export_folder = '30_11_2022'
includer_header = True
enable_zeropadding = True
zeropadding_method = 'bfill'
step = '3s'
sensors_list = [
    ('localbytes', 'power', 'living_room_tv', 'float'), 
    ('localbytes2', 'power', 'kitchen_kettle', 'float'),
    ('localbytes3', 'power', 'office_computer_setup_01', 'float'),
    ('localbytes4', 'power', 'kitchen_toaster', 'float'),
    ('localbytes5', 'power', 'kitchen_washing_machine', 'float'),
    ('localbytes7', 'power', 'office_computer_setup_02', 'float'),
    ('localbytes8', 'power', 'kitchen_fridge', 'float'),
    ('ewelink', 'th01_8dc96c24_humidity', 'living_room_humidity', 'float'),
    ('ewelink', 'th01_8dc96c24_temperature', 'living_room_temperature', 'float'),
    ('ewelink', 'th01_b0c70225_humidity', 'office_humidity', 'float'),
    ('ewelink', 'th01_b0c70225_temperature', 'office_temperature', 'float'),
    ('ewelink', 'ms01_ce61cc24_ias_zone', 'kitchen_occupancy', 'bool'),
    ('ewelink', 'ms01_72641c25_ias_zone', 'living_room_occupancy', 'bool'),
    ]
pd.options.mode.chained_assignment = None  # default='warn'
#######################

# Helper functions #
def zeropad(df, step, method):
    print('Zero-padding dataset...')
    df = df[['datetime', 'state']]
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    limit = int(180 / 3)
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=step), method=method, fill_value=0, limit=limit) 

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'datetime'}, inplace=True)
    # Create column 'ts' as unix timestamp from datetime column, make sure ts is integer   
    df['ts'] = df['datetime'].astype(np.int64) // 10**9
    df = df[['ts', 'datetime', 'state']]
    if _verbose: print(df.head(50))
    return df

def filter_data_weather(df, parameter):
    print('Filtering dataset to extract weather data...')
    df = df[['state_id', 'domain', 'entity_id', 'state', 'attributes', 'last_changed']]
    df = df.query('domain == "weather"')
    # Extract the 'attributes' as seperate coliumns as the JSON object as follows:
    # # "{""temperature"": 6.0, ""humidity"": 99, ""pressure"": 1025.5, ""wind_bearing"": 39.7, ""wind_speed"": 0.7}""
    df['attributes'] = df['attributes'].str.replace('"', '')
    df['attributes'] = df['attributes'].str.replace('{', '')
    df['attributes'] = df['attributes'].str.replace('}', '')
    df['attributes'] = df['attributes'].str.replace(' ', '')
    df['attributes'] = df['attributes'].str.replace(',,', ',')
    df['temperature'] = df['attributes'].str.split(',').str[0].str.split(':').str[1]
    df['humidity'] = df['attributes'].str.split(',').str[1].str.split(':').str[1]
    df['pressure'] = df['attributes'].str.split(',').str[2].str.split(':').str[1]
    df['wind_bearing'] = df['attributes'].str.split(',').str[3].str.split(':').str[1]
    df['wind_speed'] = df['attributes'].str.split(',').str[4].str.split(':').str[1]
    df.insert(1, 'datetime',  pd.to_datetime(df['last_changed'], format="%Y-%m-%d %H:%M:%S"))
    df.insert(2, 'ts', df.datetime.values.astype(np.int64) // 10 ** 9)
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.drop(columns = ['state_id', 'last_changed', 'entity_id', 'domain', 'attributes'])
    df.rename(columns={'state': 'weather'}, inplace=True)
    if parameter is None:
        column_names = ['datetime', 'ts', 'weather', 'temperature', 'humidity', 'pressure', 'wind_bearing', 'wind_speed']
    else:
        column_names = ['datetime', 'ts', parameter]
    df = df[pd.notna(df.weather)]
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['datetime'])
    df = df.reindex(columns=column_names)
    df.reset_index(drop=True, inplace=True)
    return df

def filter_data(df):
    print('Filtering dataset...')
    df = df[['state_id', 'domain', 'entity_id', 'state', 'last_changed']]
    df = df.query('domain != "sun" and domain != "weather" and domain != "switch" and state != "unknown" and state != "unavailable"')
    dropped_indexes = df[(df['domain'] == 'binary_sensor') & ((df['entity_id'] != 'binary_sensor.ewelink_ms01_ce61cc24_ias_zone') | (df['entity_id'] != 'binary_sensor.ewelink_ms01_72641c25_ias_zone'))].index
    # df.drop(dropped_indexes, inplace=True)
    dropped_indexes2 = df[(df['entity_id'] == 'sensor.ai_lab_plug_1') | (df['entity_id'] == 'sensor.ai_lab_plug_2')].index
    df.drop(dropped_indexes2, inplace=True)
    dropped_indexes3 = df[df['entity_id'].str.contains('abdol') | df['entity_id'].str.contains('activity_state') | df['entity_id'].str.contains('daily_energy') | df['entity_id'].str.contains('wifi') | df['entity_id'].str.contains('server')].index
    df.drop(dropped_indexes3, inplace=True)

    df = df[pd.notna(df.state)]
    df = df.drop_duplicates()

    df[['entity_type', '_entity_name']] = df['entity_id'].str.split('\.', n=1, expand=True)
    df[['entity_name', 'parameter']] = df['_entity_name'].str.split('\_', n=1, expand=True)

    df['datetime'] =  pd.to_datetime(df['last_changed'], format="%Y-%m-%d %H:%M:%S.%f")
    df = df.drop(columns = ['last_changed', 'entity_id', 'domain', '_entity_name', 'entity_type'])

    column_names = ['state_id', 'datetime', 'entity_name', 'parameter', 'state']
    df = df.reindex(columns=column_names)
    df.reset_index(drop=True, inplace=True)
    if _verbose: print(df.head(50))

    # Add unix timestamp column
    df.insert(1, 'ts', df.datetime.values.astype(np.int64) // 10 ** 9)

    return df

def export_parameters(df, plugid, parameter, filename = None, binary = False, zeropadding = False, step = '3s'):
    print("Exporting dataset as CSV with ", plugid, parameter)
    _df = df.query('entity_name == "' + plugid + '" and parameter == "'+ parameter +'"')
    _df = _df.drop(columns = ['state_id', 'entity_name', 'parameter'])

    if binary: _df['state'] = _df['state'].map({'on': 1, 'off': 0})

    if enable_zeropadding: _df = zeropad(_df, step=step, method=zeropadding_method)

    if filename is None:
        _df.to_csv(export_folder + '/' + plugid + '_' + parameter + '.csv', index=False, header=includer_header)
    else:
        _df.to_csv(export_folder + '/' + filename + '.csv', index=False, header=includer_header)
    if _verbose: print(_df.head(50)) 
    return _df

def export_weather_data(df, filename = None):
    print("Exporting weather dataset as CSV")
    if filename is None:
        df.to_csv(export_folder + '/weather.csv', index=False, header=includer_header)
    else:
        df.to_csv(export_folder + '/' + filename + '.csv', index=False, header=includer_header)

def print_compute_time_memory(time_start):
    time_elapsed = (time.perf_counter() - time_start)
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print("Computed in %5.1f s and used %5.1f MB memory" % (time_elapsed,memMb))
####################

# Main functions
def process_power():
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)

    # Import dataset master file
    time_start = time.perf_counter() # Start computational time counter
    print ('Loading data...')
    df = pd.read_csv(dataset_path)
    print('Dataset loaded from ', dataset_path)
    df = filter_data(df) # Dataset filtering
    if _verbose: print(df.head(50))
    
    # Loop through parameters list
    for i in sensors_list:
        _plugid = i[0]
        _parameter = i[1]
        _filename = i[2]
        _type = i[3]
        if _verbose: print("Currently processing", _plugid, _parameter, _filename, _type)
        if _type != 'bool':
            df2 = export_parameters(df, _plugid, _parameter, _filename, zeropadding=enable_zeropadding) # Extract specific appliance's power consumption
        else:
            df2 = export_parameters(df, _plugid, _parameter, _filename, zeropadding=enable_zeropadding, step=step, binary=True)
    print_compute_time_memory(time_start) # Calculate computational power and memory usage
    print('File saved.')

def process_weather():
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
     # Import dataset master file
    time_start = time.perf_counter() # Start computational time counter
    print ('Loading data...')
    df = pd.read_csv(dataset_path)
    print('Dataset loaded from ', dataset_path)
    # df = filter_data(df) # Dataset filtering
    df = filter_data_weather(df, 'humidity') # Dataset filtering
    if _verbose: print(df.head(50))
    export_weather_data(df, filename='humidity_outdoor') # Export weather data
    print_compute_time_memory(time_start) # Calculate computational power and memory usage
    print('File saved.')

def main():
    # process_power()
   process_weather()

main()