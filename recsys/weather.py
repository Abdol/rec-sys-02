# Imports #
from functions import *
####################

class Weather:

    def __init__(self, temp_df: pd.DataFrame, hum_df: pd.DataFrame):
        self.temp_df = temp_df
        self.hum_df = hum_df

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.tight_layout(pad=5.0)
        # Make axis labels horizontal (not rotated)
        self.temp_df.plot(ax=ax1, title='Temperature')
        self.hum_df.plot(ax=ax2, title='Humidity')
        ax1.xaxis.set_tick_params(rotation=0)
        ax2.xaxis.set_tick_params(rotation=0)
        fig.subplots_adjust(wspace=0.6)
        plt.show()

    def get_trends(self, outdoor_temp_df: pd.DataFrame, outdoor_hum_df: pd.DataFrame):
        """Write a function that compares between the indoor and outdoor temperature and humidity
        and detects when they are similar and when they are different. This will be used to
        determine whether the heater is on or off. 
        """
        # Step 1: Get the average temperature and humidity for each day
        avg_temp = self.temp_df.resample('D').mean()
        avg_hum = self.hum_df.resample('D').mean()

        # Step 2: Compare the indoor and outdoor temperature and humidity
        diff_temp = avg_temp - outdoor_temp_df
        diff_hum = avg_hum - outdoor_hum_df


        # Step 3: If the indoor and outdoor temperature and humidity are similar, then the heater is off
        # If the indoor and outdoor temperature and humidity are different, then the heater is on
        if diff_temp == 0 and diff_hum == 0:
            print('Heater is off')
        else:
            print('Heater is on')
    
        # Step 5: Plot the results
        fig, (ax1, ax2) = plt.subplots(2)
        fig.tight_layout(pad=5.0)
        # Make axis labels horizontal (not rotated)
        diff_temp.plot(ax=ax1, title='Temperature')
        diff_hum.plot(ax=ax2, title='Humidity')
        ax1.xaxis.set_tick_params(rotation=0)
        ax2.xaxis.set_tick_params(rotation=0)
        fig.subplots_adjust(wspace=0.6)
        plt.show()
        
    
        



