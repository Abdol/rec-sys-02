# Imports #
import time
import resource
import pandas as pd
from enum import Enum
if plot_enabled:
    from matplotlib import patches, pyplot as plt
    import matplotlib.dates as mdates
import numpy as np
# from sklearn.ensemble import IsolationForest
####################

# Internal constants #
verbose = False
######################

# Helper functions #
def import_data(dataset_path, dataset_limit = None):
    print(f'Importing {dataset_path}...')
    df = pd.read_csv(dataset_path)
    if dataset_limit != None: df = df.head(dataset_limit)
    return df

def prepare_data(df, column='state'):
    df = df[['datetime', column]]
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index() 
    if verbose: print(df.head(50))
    df.rename(columns={'index': 'datetime'}, inplace=True)
    return df

def postprocess_data(df, start_date = None, end_date = None):
    # crop data from specified datetime
    if start_date != None and end_date != None: 
        df = df.loc[start_date:end_date]
    return df

def export_pickle(df, path):
    print(f'Exporting pickle to {path}...')
    df.to_pickle(path)

def import_pickle(path):
    print(f'Importing pickle from {path}...')
    df = pd.read_pickle(path)
    return df

def plot_data(df):
    print('Plotting data...')
    plt.plot(df)
    plt.tight_layout()
    plt.show()

def plot_data(df, df2, label1 = 'plot1', label2 = 'plot2', fill_between = False, df1_column = 'state', df2_column = 'state'):
    print('Plotting data...')
    fig, (ax1, ax2) = plt.subplots(2)
    df[df1_column] = zero_to_nan(df[df1_column])
    ax1.step(df.index, df[df1_column], label=label1, color='blue')
    ax2.plot(df2, label=label2, color='green')
    # Scale df2 to match df
    df2 = df2 * df.max() / df2.max()
    if fill_between == True: ax1.fill_between(df2.index, df2[df2_column], label=label2, color='green', alpha=0.3)
    fig.tight_layout()
    fig.legend()
    plt.show()

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]
    
def average_amplitude(df, column='state'):
    """Calculate the average amplitude of a df"""
    return df[column].where(df[column] != 0).mean()

def print_compute_time_memory(time_start):
    time_elapsed = (time.perf_counter() - time_start)
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print("Computed in %5.1f s and used %5.1f MB memory" % (time_elapsed,memMb))
####################

# Recsys functions #
def visualize(df, df_occ, column1, column2, amp_threshold = 1000, width_threshold = 20):
    df = pd.merge_asof(df, df_occ, left_index=True, right_index=True, direction='nearest')
    df.rename(columns={'state_x': column1, 'state_y': column2}, inplace=True)
    df.fillna(0, inplace=True)
    if verbose: print(df.head(50))
    
    # Make state column = 0 when occupancy is 0
    df_no = df.copy()
    df_oc = df.copy()
    df_no.loc[df_no[column2] == 1, column1] = 0
    df_oc.loc[df_oc[column2] == 0, column1] = 0
    if verbose: print('df_no:', df_no.head(50))

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    ax1.plot(df[column1], label=column1, color='red')
    ax1.set_title(f'{column1}')
    # Plot horizontal line with amplitude = norm_amp
    # Annotate the line above with the value of norm_amp
    if norm_amp != None: 
        ax1.axhline(y=norm_amp, color='orange', linestyle='--', label='Normal amplitude')
        ax1.annotate(norm_amp, (df[column1].index[0], norm_amp), color='orange')
    ax2.plot(df[column2], label=column2, color='blue')
    ax2.set_title(f'{column2}')
    ax3.plot(df_no[column1], label=f'{column1} at no {column2}', color='orange')
    ax3.set_title(f'{column1} at no {column2}')
    ax4.plot(df_oc[column1], label=f'{column1} at {column2}', color='purple')
    ax4.set_title(f'{column1} with {column2}')
    # for i, segment in enumerate(changes):
    #     ax5.plot(segment.index, segment[column1])
    # ax5.set_title(f'{column1} segments')
    ax1.set_xlim([df[column1].index[0], df[column1].index[-1]])
    ax2.set_xlim([df[column1].index[0], df[column1].index[-1]])
    ax3.set_xlim([df[column1].index[0], df[column1].index[-1]])
    ax4.set_xlim([df[column1].index[0], df[column1].index[-1]])
    # ax5.set_xlim([df[column1].index[0], df[column1].index[-1]])
    fig.tight_layout()
    fig.legend()
    plt.show()

def extract_features(df, df_occ, column1, column2, amp_threshold = 1000, width_threshold = 20, plot = False, groupby = None, norm_amp = None):
    # Split the dataframe into segments where the value of column changes based on threshold change
    changes = split_at_change(df, column=column1, threshold=amp_threshold, width_threhold=width_threshold)
    changes_grouped = []
    if groupby != None:
        changes_grouped = split_at_change_grouped(df, groupby='1d', column=column1, threshold=amp_threshold, width_threhold=width_threshold)   
    if verbose: print(changes)

    if groupby != None:
        return changes_grouped
    return changes

def describe_features(features, sampling_freq = 3):
    """Describe the features and also measure duration of each feature in seconds"""
    for i, feature in enumerate(features):
        print('Feature', i)
        print(feature.describe())
        print('Length:', len(feature), 'units')
        print('Duration:', len(feature) * sampling_freq, 'seconds')

def occ_recs(features, column, threshold = 50, duration = 60, print_recs = True):
    """Generate recommendations based on the extracted features
    If the max value of the feature is higher than threshold and the duration is longer than duration,
    then recommend to turn off the appliance."""
    recs = []
    print('Generating occupancy-based recommendations...')
    for i, feature in enumerate(features):
        if feature[column].mean() > threshold and len(feature) > duration:
            # Explain why the recommendation is made by mentioning the feature length and mean and max consumption values
            explanation = 'Recommendation:', i, 'Turn off the appliance', 'Consumption duration:', len(feature), 's,', 'Mean power:', int(feature[column].mean()), 'W,', 'Max power:', int(feature[column].max()), 'W'
            if print_recs: print(explanation)
            recs.append((feature, 0, explanation)) # 0 = turn off
        else:
            if print_recs and verbose: print('Recommendation:', i, 'Keep on, no change required')
    return recs

def freq_recs(features, norm_freq, print_recs = True):
    recs = []
    print('Generating frequency-based recommendations...')
    # Convert features into a dataframe
    for i, (period, feature) in enumerate(features):
        if len(feature) > norm_freq:
            explanation = f"Date: {period.strftime('%Y-%m-%d')}, Recommendation: Reduce consumption frequency, Frequency: {len(feature)}, Normal frequency: {norm_freq}"
            if print_recs: print(explanation)
            recs.append((feature, 0, explanation))
        else:
            if print_recs and verbose: print('Recommendation:', i, 'Keep on, no change required')
    return recs

def amp_recs(feature, column, norm_amp, print_recs = True):
    print('Generating amplitude-based recommendations...')
    recs = []
    for i, (period, feature) in enumerate(feature):
        avg_amp = average_amplitude(feature[0], column)
        # Check if the average amplitude is higher than the normal amplitude by 20% or more
        if avg_amp > norm_amp * 1.2:
            explanation = f"Date: {period.strftime('%Y-%m-%d')}, Recommendation: Reduce consumption amplitude, Average amplitude: {avg_amp} W, Normal amplitude: {norm_amp} W"
            if print_recs: print(explanation)
            recs.append((feature, 0, explanation))
        else:
            if print_recs and verbose: print('Recommendation:', i, 'Keep on, no change required')
    return recs
####################