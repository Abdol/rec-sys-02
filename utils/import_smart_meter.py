from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
from datetime import datetime 
plt.close('all')
path = '../../data/smart_meter.csv'
start_date = '2022-08-01 00:00:00'
end_date = '2022-08-15 23:59:59'

def show_plot():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001) # Pause for interval seconds.
    input("hit[enter] to close.")
    plt.close('all') # all open plots are correctly closed after each run
    plt.show()

def import_smart_meter(path, start_date, end_date):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['timestamp (UTC)'])
    df.drop(columns=['timestamp (UTC)'], inplace=True)
    # Set datetime column as index
    df = df.set_index('datetime')
    # Sort index
    df = df.sort_index()
    # Rename state column to power
    df.rename(columns={'energyConsumption (kWh)': 'state'}, inplace=True)
    # Convert Kwh to W
    df['state'] = df['state'] / 0.5 * 1000
    # Aggregate data to daily
    df = df.resample('D').sum()
    # Crop data from specified datetime
    df = df.loc[start_date:end_date]
    return df

def main():
    df = import_smart_meter(path, start_date, end_date)

    print(df.index)
    print(df.index.min())
    print(df.index.max())

    # Plot data as bar chart and with lines colored based on power
    fig, ax = plt.subplots(figsize=(15, 8))
    # Set x-axis to datetime
    ax.set_xticks(df.index)
    # Format x-axis ticks to show only date
    ax.set_xticklabels(df.index.strftime('%d-%m'), rotation=45)
    # Set y-axis to power
    ax.set_ylim(0, df['state'].max())
    # Label axes
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Power (W)')
    # Plot data as bar chart
    ax.bar(df.index, df['state'], width=0.018, color='blue', edgecolor='grey', linewidth=0.5)
    show_plot()

main()