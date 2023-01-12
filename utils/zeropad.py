import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
_verbose = False
dataset_path = '../appliances_power/kitchen_kettle.csv'
export_path = '../appliances_power/kitchen_kettle_zp.csv'


def zeropad(df, step):
    print('Zero-padding dataset...')
    df = df[['datetime', 'state']]
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=step), method='bfill', fill_value=0)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'datetime'}, inplace=True)
    # Create column 'ts' as unix timestamp from datetime column, make sure ts is integer   
    df['ts'] = df['datetime'].astype(np.int64) // 10**9
    df = df[['ts', 'datetime', 'state']]
    if _verbose: print(df.head(50))
    return df

def o1(df, plot=False):
    print(df.head())
    print('first index', df.index[0])
    print('last index', df.index[-1])
    df_zp = zeropad(df, '3s')
    print(df_zp.head())
    df_zp.to_csv(export_path, index=False)
    if plot:
        plt.plot(df_zp['datetime'], df_zp['state'], label='zeropad')
    return df_zp


def o2(df):
    # plot the two datasets, x-axis is datetime, y-axis is state
    # show datetime on x-axis
    df = df[['datetime', 'state']]
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index() 
    return df  

def main():
    _df = pd.read_csv(dataset_path)
    df_zp = o1(_df, plot=False)
    df = o2(_df)
    # Plot the two dataframes on the same plot 
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(df, label='original')
    ax2.plot(df, label='original')
    ax2.plot(df_zp['datetime'], df_zp['state'], label='zeropad')
    plt.legend()
    plt.show()


main()




