# Imports #
import time
from constants import *
import recsys as rs
from functions import *
from recsys.recommender import RecommendationType
plt.rcParams.update({'font.size': plot_font_size, 'figure.figsize': plot_size})
####################

print('Starting...')
# Import pickles #
start = time.perf_counter()
df_temp_outdoor = import_pickle(pickle_path_weather_temp)
df_hum_outdoor = import_pickle(pickle_path_weather_hum)
df_occ_living_room = import_pickle(pickle_path_occ_living_room)
df_occ_kitchen = import_pickle(pickle_path_occ_kitchen)
df_temp_living_room = import_pickle(pickle_path_temp_living_room)
df_temp_office = import_pickle(pickle_path_temp_office)
df_hum_living_room = import_pickle(pickle_path_hum_living_room)
df_hum_office = import_pickle(pickle_path_hum_office)
kettle_df = import_pickle(pickle_path_kettle)
tv_df = import_pickle(pickle_path_tv)
toaster_df = import_pickle(pickle_path_toaster)
fridge_df = import_pickle(pickle_path_fridge)
washing_machine_df = import_pickle(pickle_path_washing_machine)
computer1_df = import_pickle(pickle_path_computer1)
computer2_df = import_pickle(pickle_path_computer2)
print_compute_time_memory(start)
####################

# Data imports #
# start = time.perf_counter()
# weather_temp_df = postprocess_data(prepare_data(import_data(dataset_path_weather), 'temperature'), start_date=start_date, end_date=end_date)
# weather_hum_df = postprocess_data(prepare_data(import_data(dataset_path_weather), 'humidity'), start_date=start_date, end_date=end_date)
# df_occ_living_room = postprocess_data(prepare_data(import_data(dataset_path_occ_living_room)), start_date=start_date, end_date=end_date)
# df_occ_kitchen = postprocess_data(prepare_data(import_data(dataset_path_occ_kitchen)), start_date=start_date, end_date=end_date)
# df_temp_living_room = postprocess_data(prepare_data(import_data(dataset_path_temp_living_room)), start_date=start_date, end_date=end_date)
# df_temp_office = postprocess_data(prepare_data(import_data(dataset_path_temp_office)), start_date=start_date, end_date=end_date)
# df_hum_living_room = postprocess_data(prepare_data(import_data(dataset_path_hum_living_room)), start_date=start_date, end_date=end_date)
# df_hum_office = postprocess_data(prepare_data(import_data(dataset_path_hum_office)), start_date=start_date, end_date=end_date)
# df_temp_outdoor = postprocess_data(prepare_data(import_data(dataset_path_temp_outdoor), 'temperature'), start_date=start_date, end_date=end_date)
# df_hum_outdoor = postprocess_data(prepare_data(import_data(dataset_path_hum_outdoor), 'humidity'), start_date=start_date, end_date=end_date)
# kettle_df = postprocess_data(prepare_data(import_data(dataset_path_kettle)), start_date=start_date, end_date=end_date)
# tv_df = postprocess_data(prepare_data(import_data(dataset_path_tv)), start_date=start_date, end_date=end_date)
# toaster_df = postprocess_data(prepare_data(import_data(dataset_path_toaster)), start_date=start_date, end_date=end_date)
# fridge_df = postprocess_data(prepare_data(import_data(dataset_path_fridge)), start_date=start_date, end_date=end_date)
# washing_machine_df = postprocess_data(prepare_data(import_data(dataset_path_washing_machine)), start_date=washing_machine_start_date, end_date=washing_machine_end_date)
# computer1_df = postprocess_data(prepare_data(import_data(dataset_path_computer1)), start_date=start_date, end_date=end_date)
# computer2_df = postprocess_data(prepare_data(import_data(dataset_path_computer2)), start_date=start_date, end_date=end_date)
# print_compute_time_memory(start)
# start = time.perf_counter()
# export_pickle(df_occ_living_room, pickle_path_occ_living_room)
# export_pickle(df_occ_kitchen, pickle_path_occ_kitchen)
# export_pickle(df_temp_living_room, pickle_path_temp_living_room)
# export_pickle(df_temp_office, pickle_path_temp_office)
# export_pickle(df_hum_living_room, pickle_path_hum_living_room)
# export_pickle(df_hum_office, pickle_path_hum_office)
# export_pickle(kettle_df, pickle_path_kettle)
# export_pickle(tv_df, pickle_path_tv)
# export_pickle(toaster_df, pickle_path_toaster)
# export_pickle(fridge_df, pickle_path_fridge)
# export_pickle(washing_machine_df, pickle_path_washing_machine)
# export_pickle(computer1_df, pickle_path_computer1)
# export_pickle(computer2_df, pickle_path_computer2)
# export_pickle(df_temp_outdoor, pickle_path_weather_temp)
# export_pickle(df_hum_outdoor, pickle_path_weather_hum)
# print_compute_time_memory(start)
####################

# Appliances #
start = time.perf_counter()
print('Creating appliance objects...')
kettle = rs.Appliance(
    df=kettle_df, 
    label='kettle', 
    amp_threshold=2500, 
    width_threshold=20, 
    norm_amp=3000,
    norm_freq=3,
    groupby='1d')
tv = rs.Appliance(
    df=tv_df, 
    label='tv', 
    amp_threshold=40, 
    width_threshold=10, 
    norm_amp=65,
    norm_freq=1,
    groupby='1d',
    df_occ=df_occ_living_room)
toaster = rs.Appliance(
    df=toaster_df, 
    label='toaster', 
    amp_threshold=500, 
    width_threshold=20, 
    # norm_amp=600,
    norm_freq=2,
    groupby='1d')
fridge = rs.Appliance(
    df=fridge_df,
    label='fridge',
    amp_threshold=100,
    width_threshold=10,
    norm_freq=8,
    groupby='1d')
washing_machine = rs.Appliance(
    df=washing_machine_df,
    label='washing_machine',
    amp_threshold=100,
    width_threshold=10,
    norm_freq=2,
    groupby='7d')
computer1 = rs.Appliance(
    df=computer1_df,
    label='computer1',
    amp_threshold=30,
    width_threshold=20,
    norm_amp=5,
    groupby='1d')
computer2 = rs.Appliance(
    df=computer2_df,
    label='computer2',
    amp_threshold=30,
    width_threshold=20,
    norm_amp=5,
    groupby='1d')
print_compute_time_memory(start)
####################

def plot():
    # Plot appliances
    tv.plot()
    kettle.plot()
    toaster.plot()
    fridge.plot()
    washing_machine.plot()
    computer1.plot()
    computer2.plot()

def rec():
    print('Generating recommendations between {} and {}...'.format(start_date, end_date))
    # Instantiate a recommender
    rec_tv = rs.Recommender(app=tv, config={RecommendationType.AMP, RecommendationType.FREQ, RecommendationType.OCC})
    rec_toaster = rs.Recommender(app=toaster, config={RecommendationType.AMP, RecommendationType.FREQ})
    rec_kettle = rs.Recommender(app=kettle, config={RecommendationType.FREQ})
    rec_fridge = rs.Recommender(app=fridge, config={RecommendationType.AMP})
    rec_washing_machine = rs.Recommender(app=washing_machine, config={RecommendationType.FREQ, RecommendationType.AMP})
    rec_computer1 = rs.Recommender(app=computer1, config={RecommendationType.AMP})
    rec_computer2 = rs.Recommender(app=computer2, config={RecommendationType.AMP})
    
    # Generate recommendations
    recs_tv = rec_tv.generate()
    recs_toaster = rec_toaster.generate()
    recs_kettle = rec_kettle.generate()
    recs_fridge = rec_fridge.generate()
    recs_washing_machine = rec_washing_machine.generate()
    recs_computer1 = rec_computer1.generate()
    recs_computer2 = rec_computer2.generate()
    
    # Print recommendations
    recs_explained = {
        'tv:':[str(row.relevance) + " " + row.explanation for row in recs_tv],
        'toaster:':[str(row.relevance) + " " + row.explanation for row in recs_toaster],
        'kettle:':[str(row.relevance) + " " + row.explanation for row in recs_kettle],
        'fridge:':[str(row.relevance) + " " + row.explanation for row in recs_fridge],
        'washing_machine:':[str(row.relevance) + " " + row.explanation for row in recs_washing_machine],
        'computer1:':[str(row.relevance) + " " + row.explanation for row in recs_computer1],
        'computer2:':[str(row.relevance) + " " + row.explanation for row in recs_computer2]
    }; print('\nRecommendations:')
    for key, value in recs_explained.items():
        print(key, *value, sep='\n')
    
    # Plot recommendations
    # rec_tv.plot()
    # rec_toaster.plot()
    # rec_kettle.plot()
    # rec_fridge.plot()
    # rec_washing_machine.plot()
    # rec_computer1.plot()
    # rec_computer2.plot()

    # Compute savings
    print('\nSavings:')
    savings_tv = rec_tv.savings(tariff)
    if savings_tv is not None: print('TV:', savings_tv / 100, '£')
    savings_toaster = rec_toaster.savings(tariff)
    if savings_toaster is not None: print('Toaster:', savings_toaster / 100, '£')
    savings_kettle = rec_kettle.savings(tariff)
    if savings_kettle is not None: print('Kettle:', savings_kettle / 100, '£')
    savings_fridge = rec_fridge.savings(tariff)
    if savings_fridge is not None: print('Fridge:', savings_fridge / 100, '£')
    savings_washing_machine = rec_washing_machine.savings(tariff)
    if savings_washing_machine is not None: print('Washing Machine:', savings_washing_machine / 100, '£')
    savings_computer1 = rec_computer1.savings(tariff)
    if savings_computer1 is not None: print('Computer 1:', savings_computer1 / 100, '£')
    savings_computer2 = rec_computer2.savings(tariff)
    if savings_computer2 is not None: print('Computer 2:', savings_computer2 / 100, '£\n')

    # Evaluate recommendations
    eval_tv = rs.Evaluator(rec_tv, rec_tv.y_pred)
    eval_toaster = rs.Evaluator(rec_toaster, rec_toaster.y_pred)
    eval_kettle = rs.Evaluator(rec_kettle, rec_kettle.y_pred)
    eval_fridge = rs.Evaluator(rec_fridge, rec_fridge.y_pred)
    eval_washing_machine = rs.Evaluator(rec_washing_machine, rec_washing_machine.y_pred)
    eval_computer1 = rs.Evaluator(rec_computer1, rec_computer1.y_pred)
    eval_computer2 = rs.Evaluator(rec_computer2, rec_computer2.y_pred)

    print('\nEvaluation Reports:')
    print('TV:', eval_tv.report())
    print('Toaster:', eval_toaster.report())
    print('Kettle:', eval_kettle.report())
    print('Fridge:', eval_fridge.report())
    print('Washing Machine:', eval_washing_machine.report())
    print('Computer Setup 1:', eval_computer1.report())
    print('Computer Setup 2:', eval_computer2.report())
    # eval.confusion_matrix()

def rec_household():
    print('Generating recommendations between {} and {}...'.format(start_date, end_date))
    # Instantiate a recommender
    rec_tv = rs.Recommender(app=tv, config={RecommendationType.AMP, RecommendationType.FREQ, RecommendationType.OCC})
    rec_toaster = rs.Recommender(app=toaster, config={RecommendationType.AMP, RecommendationType.FREQ})
    rec_kettle = rs.Recommender(app=kettle, config={RecommendationType.FREQ})
    rec_fridge = rs.Recommender(app=fridge, config={RecommendationType.AMP})
    rec_washing_machine = rs.Recommender(app=washing_machine, config={RecommendationType.FREQ, RecommendationType.AMP})
    rec_computer1 = rs.Recommender(app=computer1, config={RecommendationType.AMP})
    rec_computer2 = rs.Recommender(app=computer2, config={RecommendationType.AMP})

    house = rs.Building([rec_tv, rec_toaster, rec_kettle, rec_fridge, rec_washing_machine, rec_computer1, rec_computer2], tariff)
    house.generate_recs()
    print(house.individial_report())
    print('Household report:', house.report())
    # house.plot_recs()

def weather():
    living_weather = rs.Weather(df_temp_living_room, df_hum_living_room, df_temp_outdoor, df_hum_outdoor)
    # office_weather = rs.Weather(df_temp_office, df_hum_office)
    # living_weather.plot()
    # diff = living_weather.trends()
    # print(diff)
    # living_weather.plot()
    # office_weather.plot()
    heater = rs.VirtualAppliance(weather=living_weather, amp_threshold=1000, norm_amp=2000, groupby='1d')
    

def main():
    # Record computation time
    start = time.perf_counter()
    rec_household()
    # rec()
    # plot()
    # weather()
    print_compute_time_memory(start)

main()

# Experiments
def acf():
    from statsmodels.graphics.tsaplots import plot_acf
    print('Plotting ACF...')
    plot_acf(kettle_df)
    plt.show()   

def stats():
    from statsmodels.tsa.stattools import adfuller
    print('Testing stationarity of the time series data...')
    results = adfuller(kettle_df)
    print('ADF Statistic: %f' % results[0])
    print('p-value: %f' % results[1])
    print('Critical Values:')
    for key, value in results[4].items():
        print('\t%s: %.3f' % (key, value))

def seasonal_decompose():
    from statsmodels.tsa.seasonal import seasonal_decompose
    print('Decomposing time series data...')
    result = seasonal_decompose(kettle_df, model='additive', period=10)
    result.plot()
    plt.show()

def lstm():
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    import numpy as np

    # power_consumption is the time series data for an appliance
    # split the data into training and test sets
    train_data = kettle_df[:int(len(kettle_df)*0.8)]
    test_data = kettle_df[int(len(kettle_df)*0.8):]

    # reshape the data for the LSTM model
    X_train = np.reshape(train_data, (len(train_data), 1))
    X_test = np.reshape(test_data, (len(test_data), 1))

    # create the LSTM modelcon
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # train the model
    model.fit(X_train, train_data, epochs=100, batch_size=1, verbose=2)

    # make predictions on the test data
    predictions = model.predict(X_test)
    print(predictions)

def isolation_forest():
    from sklearn.ensemble import IsolationForest
    import numpy as np
    global kettle_df
    # fit the model
    clf = IsolationForest(random_state=0).fit(kettle_df)

    # predict the usage frequency
    kettle_df['freq'] = [1 if pred == -1 else 0 for pred in clf.predict(kettle_df)]


    # Normalize kettle_df['state'] to 0 and 1

    # Compute average usage frequency per day as a float
    # daily_average_freq = kettle_df['freq'].resample('1d').sum().mean()
    # print('Average usage frequency per day: {}'.format(daily_average_freq))

    freq = rs.Appliance.split_at_change_grouped(kettle_df, '1d', 0, 10, 'freq') 
    average = round(np.mean(np.array([len(f) for p, f in freq])))
    
    print(average)
    for p, f in freq:
        print(p, len(f))

    kettle_df['state'] = (kettle_df['state'] - kettle_df['state'].min()) / (kettle_df['state'].max() - kettle_df['state'].min()) * 2
    plt.plot(kettle_df.index, kettle_df['freq'])
    plt.plot(kettle_df.index, kettle_df['state'])
    plt.show()
