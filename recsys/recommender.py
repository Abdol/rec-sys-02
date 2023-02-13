# Imports #
from functions import *
from .appliance import Appliance
from .recommendation import Recommendation, RecommendationType
######################
verbose = False # TODO: use logging and argpass instead

# Class definition #
class Recommender:
    def __init__(self, app: Appliance, config = {RecommendationType.AMP, RecommendationType.FREQ, RecommendationType.OCC}):
        self._app = app
        self._recs = []
        self.config = config

    @property
    def app(self):
        return self._app

    @property
    def recs(self):
        if len(self._recs) == 0:
            # Throw error because recs is empty
            # raise Exception('No recommendations generated yet. Call generate() first.')
            return []
        return self._recs

    @property
    def y_pred(self):
        if len(self._recs) == 0:
            # # Throw error because recs is empty
            # raise Exception('No recommendations generated yet. Call generate() first.')
            # # TODO: account when generate() is called but no recs are generated
            return []
        return [rec.action for rec in self._recs]

    def amp(self):
        if verbose: print('Generating amplitude-based recommendations...')
        recs = []
        features = self.app.features
        norm_amp = self.app.norm_amp
        column = self.app.column
        df = self.app.df

        for i, (period, feature) in enumerate(features):
            avg_amp = df[column].where(feature[0][column] != 0).mean()
            # Check if the average amplitude is higher than the normal amplitude by 20% or more
            if avg_amp > norm_amp * 1.2:
                rec = Recommendation(
                    datetime=period,
                    duration=len(feature) * self.app.sample_rate,
                    app=self.app,
                    action=1,
                    # explanation=f"Amplitude Recommendation: Reduce consumption amplitude, Average amplitude: {avg_amp} W, Normal amplitude: {norm_amp} W",
                    # Format avg_amp to 2 decimal places
                    explanation=f"{avg_amp:.2f} W",
                    type=RecommendationType.AMP
                )
                recs.append(rec)
        return recs

    def freq(self):
        recs = []
        features = self.app.features
        norm_freq = self.app.norm_freq
        if norm_freq == None:
            raise Exception('No normal usage frequency found.')
        if verbose: print('Generating frequency-based recommendations...')
        # Convert features into a dataframe
        for i, (period, feature) in enumerate(features):
            f = len(feature)
            if f > norm_freq:
                rec = Recommendation(
                    datetime=period,
                    duration=f * self.app.width_threshold * self.app.sample_rate,
                    app=self.app,
                    action=1,
                    # explanation=f"Frequency Recommendation: Reduce consumption frequency, Frequency: {len(feature)}, Normal frequency: {norm_freq}",
                    explanation=f"{f}",
                    type=RecommendationType.FREQ
                )
                recs.append(rec)
        return recs

    def occ(self):
        """Generate recommendations based on the extracted features
        If the max value of the feature is higher than threshold and the duration is longer than duration,
        then recommend to turn off the appliance."""
        recs = []
        df = self.app.df
        df_occ = self.app.df_occ
        column = self.app.column
        threshold = self.app.norm_amp
        groupby = self.app.groupby
        duration = self.app.width_threshold
        column1 = self.app.column
        column2 = 'occupancy'

        df = pd.merge_asof(df, df_occ, left_index=True, right_index=True, direction='nearest')
        df.rename(columns={'state_x': column1, 'state_y': column2}, inplace=True)
        df.fillna(0, inplace=True)

        # Make state column = 0 when occupancy is 0
        df_no_occ = df.copy()
        df_occ = df.copy()
        df_no_occ.loc[df_no_occ[column2] == 1, column1] = 0
        df_occ.loc[df_occ[column2] == 0, column1] = 0

        features = self._app.split_at_change_grouped(df_no_occ, column=column1, threshold=threshold, width_threhold=duration, groupby=groupby)

        if verbose: print('Generating occupancy-based recommendations...')
        for i, (period, feature) in enumerate(features):
            mean_power = feature[0][column].mean()
            duration = len(feature) * self.app.sample_rate
            if mean_power > threshold and len(feature) > duration:
                rec = Recommendation(
                    datetime=period,
                    duration=duration,
                    app=self.app,
                    action=1, # 1: change state, 0: don't change
                    # explanation=f"Occupancy Recommendation: Turn off the appliance, Consumption duration: {len(feature)} s, Mean power: {int(feature[0][column].mean())} W, Max power: {int(feature[0][column].max())} W",
                    explanation=f"{duration} s, {mean_power} W",
                    type=RecommendationType.OCC
                )
                recs.append(rec) 
        return recs

    def generate(self):
        if RecommendationType.FREQ in self.config: self._recs += self.freq()
        if RecommendationType.AMP in self.config: self._recs += self.amp()
        if RecommendationType.OCC in self.config: self._recs += self.occ()

        # Sort recommendations by relevance in descending order
        self._recs.sort(key=lambda x: x.relevance, reverse=True)
        return self._recs

    def plot(self):
        if len(self._recs) == 0:
            # Throw warning because recs is empty
            print('No recommendations generated yet. Call generate() first.')
            return
        fig, ax = plt.subplots()
        ax.plot(self.app.df.index, self.app.df[self.app.column], color='grey', label=self.app.label)

        for rec in self._recs:
            # Plot recommendation and color them according to their type (amp, freq, occ)
            if rec.type == RecommendationType.AMP:
                ax.axvline(x=rec.datetime, ymin=0.9, ymax=0.95, color='red', linestyle='-')
                # ax.axvspan(rec.datetime, rec.datetime + pd.Timedelta(seconds=rec.duration) * 3, color='red', alpha=0.6)
                # Annotate text to the left side of the line
                ax.annotate(rec.explanation, xy=(rec.datetime, rec.app.df[self.app.column].max()), xytext=(rec.datetime, rec.app.df[self.app.column].max()))
            elif rec.type == RecommendationType.FREQ:
                x = rec.datetime + pd.Timedelta(hours=1)
                ax.axvline(x=x, color='blue', ymin=0.9, ymax=0.95, linestyle='-')
                ax.annotate(rec.explanation, xy=(x, rec.app.df[self.app.column].max()), xytext=(x, rec.app.df[self.app.column].max() + 1))
            elif rec.type == RecommendationType.OCC:
                ax.axvspan(rec.datetime, rec.datetime + pd.Timedelta(seconds=rec.duration), color='green', alpha=0.8)
                ax.annotate(rec.explanation, xy=(rec.datetime, rec.app.df[self.app.column].max()), xytext=(rec.datetime, rec.app.df[self.app.column].max() + 2))
        # Manually create legend
        ax.legend(handles=[
            patches.Patch(color='red', label='Amplitude'),
            patches.Patch(color='blue', label='Frequency'),
            patches.Patch(color='green', label='Occupancy')
        ])
        # Plot normal amplitude
        ax.axhline(y=self.app.norm_amp, color='black', linestyle='--', label='Normal amplitude')
        ax.annotate(f"An: {self.app.norm_amp:.2f} W", xy=(self.app.df.index[0], self.app.norm_amp), xytext=(self.app.df.index[0], self.app.norm_amp + 1))
        # Plot normal frequency
        ax.axhline(y=self.app.norm_freq, color='black', linestyle='--', label='Normal frequency')
        ax.annotate(f"fn: {self.app.norm_freq}", xy=(self.app.df.index[0], self.app.norm_freq), xytext=(self.app.df.index[0], self.app.norm_freq + 1))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Recommendations for \'{}\''.format(self.app.label))
        plt.show()
####################