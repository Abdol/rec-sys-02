# Imports #
import time
import os
import psutil
from constants import *
import recsys as rs
from functions import *
####################

# Data imports #
df_occ_kitchen = postprocess_data(prepare_data(import_data(dataset_path_occ_kitchen)), start_date=start_date, end_date=end_date)
kettle_df = postprocess_data(prepare_data(import_data(dataset_path_kettle)), start_date=start_date, end_date=end_date)
tv_df = postprocess_data(prepare_data(import_data(dataset_path_tv)), start_date=start_date, end_date=end_date)
toaster_df = postprocess_data(prepare_data(import_data(dataset_path_toaster)), start_date=start_date, end_date=end_date)
fridge_df = postprocess_data(prepare_data(import_data(dataset_path_fridge)), start_date=start_date, end_date=end_date)
washing_machine_df = postprocess_data(prepare_data(import_data(dataset_path_washing_machine)), start_date=washing_machine_start_date, end_date=washing_machine_end_date)
computer1_df = postprocess_data(prepare_data(import_data(dataset_path_computer1)), start_date=start_date, end_date=end_date)
computer2_df = postprocess_data(prepare_data(import_data(dataset_path_computer2)), start_date=start_date, end_date=end_date)

####################

# Appliances #
kettle = rs.Appliance(
    df=kettle_df, 
    column='state', 
    amp_threshold=2500, 
    width_threshold=20, 
    norm_amp=3000,
    norm_freq=3,
    groupby='1d')
tv = rs.Appliance(
    df=tv_df, 
    column='state', 
    amp_threshold=40, 
    width_threshold=10, 
    norm_amp=50,
    norm_freq=1,
    groupby='1d',
    df_occ=df_occ_kitchen)
toaster = rs.Appliance(
    df=toaster_df, 
    column='state', 
    amp_threshold=500, 
    width_threshold=20, 
    norm_amp=600,
    norm_freq=2,
    groupby='1d')
fridge = rs.Appliance(
    df=fridge_df,
    column='state',
    amp_threshold=100,
    width_threshold=10,
    groupby='1d')
washing_machine = rs.Appliance(
    df=washing_machine_df,
    column='state',
    amp_threshold=100,
    width_threshold=10,
    norm_freq=2,
    groupby='7d')
computer1 = rs.Appliance(
    df=computer1_df,
    column='state',
    amp_threshold=30,
    width_threshold=20,
    groupby='1d')
computer2 = rs.Appliance(
    df=computer2_df,
    column='state',
    amp_threshold=30,
    width_threshold=20,
    groupby='1d')
####################

def rec():
    # Plot appliances
    # tv.plot()
    # kettle.plot()
    # toaster.plot()
    # fridge.plot()
    # washing_machine.plot()
    # computer1.plot()
    # computer2.plot()
   
    # Instantiate a recommender
    rec_tv = rs.Recommender(app=tv)
    rec_toaster = rs.Recommender(app=toaster)
    rec_kettle = rs.Recommender(app=kettle)
    rec_fridge = rs.Recommender(app=fridge)
    rec_washing_machine = rs.Recommender(app=washing_machine)
    rec_computer1 = rs.Recommender(app=computer1)
    rec_computer2 = rs.Recommender(app=computer2)
    
    # Generate recommendations
    recs_tv = rec_tv.generate(freq=True, amp=True, occ=True)
    recs_toaster = rec_toaster.generate(freq=True, amp=True, occ=False)
    recs_kettle = rec_kettle.generate(freq=True, amp=True, occ=False)
    recs_fridge = rec_fridge.generate(freq=True, amp=True, occ=False)
    recs_washing_machine = rec_washing_machine.generate(freq=True, amp=True, occ=False)
    recs_computer1 = rec_computer1.generate(freq=True, amp=True, occ=False)
    recs_computer2 = rec_computer2.generate(freq=True, amp=True, occ=False)
    
    # Print recommendations
    recs_explained = {
        'tv':[str(row.relevance) + " " + row.explanation for row in recs_tv],
        'toaster':[str(row.relevance) + " " + row.explanation for row in recs_toaster],
        'kettle':[str(row.relevance) + " " + row.explanation for row in recs_kettle],
        'fridge':[str(row.relevance) + " " + row.explanation for row in recs_fridge],
        'washing_machine':[str(row.relevance) + " " + row.explanation for row in recs_washing_machine],
        'computer1':[str(row.relevance) + " " + row.explanation for row in recs_computer1],
        'computer2':[str(row.relevance) + " " + row.explanation for row in recs_computer2]
    }; print('Recommendations:')
    for key, value in recs_explained.items():
        print(key, *value, sep='\n')
    
    # Evaluate recommendations
    eval_tv = rs.Evaluator(rec_tv, rec_tv.y_pred)
    eval_toaster = rs.Evaluator(rec_toaster, rec_toaster.y_pred)
    eval_kettle = rs.Evaluator(rec_kettle, rec_kettle.y_pred)
    eval_fridge = rs.Evaluator(rec_fridge, rec_fridge.y_pred)
    eval_washing_machine = rs.Evaluator(rec_washing_machine, rec_washing_machine.y_pred)
    eval_computer1 = rs.Evaluator(rec_computer1, rec_computer1.y_pred)
    eval_computer2 = rs.Evaluator(rec_computer2, rec_computer2.y_pred)

    print('Evaluation Reports:')
    print('TV:', eval_tv.report())
    print('Toaster:', eval_toaster.report())
    print('Kettle:', eval_kettle.report())
    print('Fridge:', eval_fridge.report())
    print('Washing Machine:', eval_washing_machine.report())
    print('Computer Setup 1:', eval_computer1.report())
    print('Computer Setup 2:', eval_computer2.report())
    # eval.confusion_matrix()

def main():
    # Record computation time
    start = time.perf_counter()
    rec()
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
