# Imports #
from functions import *
from constants import *
####################

class Weather:
    def __init__(self, temp_df: pd.DataFrame, hum_df: pd.DataFrame, temp_df_outdoor: pd.DataFrame, hum_df_outdoor: pd.DataFrame):
        self.temp_df = temp_df
        self.hum_df = hum_df
        self.temp_df_outdoor = temp_df_outdoor
        self.hum_df_outdoor = hum_df_outdoor
        self._temp_df_daily = self.temp_df.resample('H').mean()
        self._hum_df_daily = self.hum_df.resample('H').mean()
        self._temp_df_outdoor_daily = self.temp_df_outdoor.resample('H').mean().interpolate(method='linear')
        self._hum_df_outdoor_daily = self.hum_df_outdoor.resample('H').mean().interpolate(method='linear')

    def plot(self, daily: bool = False):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.tight_layout(pad=5.0)
        # Make axis labels horizontal (not rotated)
        if daily:
            self._temp_df_daily.plot(ax=ax1, title='Temperature')
            self._temp_df_outdoor_daily.plot(ax=ax1, title='Temperature')
            self._hum_df_daily.plot(ax=ax2, title='Humidity')
            self._hum_df_outdoor_daily.plot(ax=ax2, title='Humidity')
            ax1.legend(['Indoor', 'Outdoor'])
            ax2.legend(['Indoor', 'Outdoor'])
        else:
            self.temp_df.plot(ax=ax1, title='Temperature')
            self.temp_df_outdoor.plot(ax=ax1, title='Temperature')
            self.hum_df.plot(ax=ax2, title='Humidity')
            self.hum_df_outdoor.plot(ax=ax2, title='Humidity')
        ax1.xaxis.set_tick_params(rotation=0)
        ax2.xaxis.set_tick_params(rotation=0)
        fig.subplots_adjust(wspace=0.6)
        plt.show()

    def import_smart_meter(self, path, start_date, end_date):
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
        # df = df.resample('D').sum()
        # Crop data from specified datetime
        df = df.loc[start_date:end_date]
        return df

    def diff(self, plot: bool = False):
        print('Getting trends...')
        # self.plot(daily=True)
        # Get differntial of temperature and humidity
        temp_diff = self._temp_df_daily.diff()
        hum_diff = self._hum_df_daily.diff()
        temp_outdoor_diff = self._temp_df_outdoor_daily.diff()
        hum_outdoor_diff = self._hum_df_outdoor_daily.diff()
        # Compare indoor and outdoor temperature diffs
        # Where indoor temp is higher than outdoor temp, the difference is positive
        # Where indoor temp is lower than outdoor temp, the difference is negative
        temp_diff['diff'] = temp_diff['state'] - temp_outdoor_diff['temperature']
        hum_diff['diff'] = hum_diff['state'] - hum_outdoor_diff['humidity']
        # Extract the days where the difference is more than 1 degree
        temp_diff = temp_diff.loc[temp_diff['diff'] > 1]
        hum_diff = hum_diff.loc[hum_diff['diff'] > 1]
        if plot:
            # Plot the difference
            fig, (ax1, ax2) = plt.subplots(2)
            fig.tight_layout(pad=5.0)
            temp_diff['diff'].plot(ax=ax1, title='Temperature')
            hum_diff['diff'].plot(ax=ax2, title='Humidity')
            ax1.xaxis.set_tick_params(rotation=0)
            ax2.xaxis.set_tick_params(rotation=0)
            fig.subplots_adjust(wspace=0.6)
            plt.show()
        return temp_diff

        


        



