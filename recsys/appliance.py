from functions import *


# Class definition #
class Appliance:
    def __init__(
        self, 
        df: pd.DataFrame, 
        label: str, 
        amp_threshold: float, 
        width_threshold: float, 
        groupby: str,
        norm_freq: float = None,
        norm_amp: float = None,
        df_occ: pd.DataFrame = None,
        sample_rate: int = 3
    ):
        self.label = label
        self._column = 'state'
        self._amp_threshold = amp_threshold
        self._width_threshold = width_threshold
        self._groupby = groupby
        self._df = df
        self._df_occ = df_occ
        self._sample_rate = sample_rate
        self._features = self.__analyze()
        self._norm_freq = norm_freq if norm_freq else self.compute_average_freq()
        self._norm_amp = norm_amp if norm_amp else self.compute_average_amp()

    @staticmethod
    def split_at_change_grouped(df, groupby, threshold, width_threhold, column):
        if verbose: print('Splitting data into grouped segments...')
        grouped_segments = []
        df_grouped = df.groupby(pd.Grouper(freq=groupby))
        for period, df_period in df_grouped:
            changes = df_period[column].diff().fillna(0).abs().gt(threshold).cumsum()
            segments = [df_period.loc[changes == i] for i in changes.unique()]
            segments = [segment for segment in segments if len(segment) > width_threhold]
            segments = [segment for segment in segments if segment[column].max() > threshold]
            segments = [segment[segment[column] > 0] for segment in segments]
            if len(segments) > 0: grouped_segments.append((period, segments))
        return grouped_segments

    def compute_average_amp(self):
        """Compute the average amplitude of the appliance"""
        df = self._df
        column = self._column
        
        # Method 1
        # amp1 = 0
        # # Get all non-zero segments
        # segments = [segment for segment in self._features[0][1] if len(segment) > 0]
        # # Compute the average amplitude of the segments
        # amp1 = np.mean([segment[column].mean() for segment in segments])

        # Method 2
        amp2 = 0
        features = self._features
        amp2 = np.mean([np.mean([segment[column].mean() for segment in feature[1]]) for feature in features])
        return amp2

    def compute_average_freq(self):
        """Compute the average frequency of the appliance"""
        df = self.df
        clf = IsolationForest(random_state=0).fit(df)
        df['freq'] = [1 if pred == -1 else 0 for pred in clf.predict(df)]
        freq = self.split_at_change_grouped(df, self.groupby, 0, self.width_threshold, 'freq') 
        average = round(np.mean(np.array([len(f) for p, f in freq])))
        # kettle_df['state'] = (kettle_df['state'] - kettle_df['state'].min()) / (kettle_df['state'].max() - kettle_df['state'].min()) * 2
        # plt.plot(kettle_df.index, kettle_df['freq'])
        # plt.plot(kettle_df.index, kettle_df['state'])
        # TODO: change parameters to adjust to varying appliances
        # TODO: Consider returning an average array
        return average

    @property
    def df(self):
        return self._df
    
    @property
    def df_occ(self):
        return self._df_occ

    @property
    def column(self):
        return self._column

    @property
    def amp_threshold(self):
        return self._amp_threshold

    @property
    def width_threshold(self):
        return self._width_threshold

    @property
    def norm_amp(self):
        return self._norm_amp

    @property
    def norm_freq(self):
        return self._norm_freq

    @property
    def groupby(self):
        return self._groupby

    @property
    def features(self):
        return self._features

    @property
    def sample_rate(self):
        return self._sample_rate

    def __analyze(self):
        return self.split_at_change_grouped(self._df, groupby=self._groupby, column=self._column, threshold=self._amp_threshold, width_threhold=self._width_threshold) 

    def plot(self):
        label = self.label
        column1 = self._column
        norm_amp = self._norm_amp
        norm_freq = self._norm_freq
        df = self._df
        features = self._features
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(df, label=column1, color='red')
        ax1.set_title(f'{label}')
        ax1.set_ylabel('Power Consumption (W)')
        ax1.set_xlabel('Date')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        if norm_amp != None: 
            amp = self.compute_average_amp()
            # ax1.axhline(y=norm_amp, color='orange', linestyle='--', label='Normal amplitude')
            ax1.axhline(y=amp, color='blue', linestyle='--', label='Appliance Average Power Consumption (AAPC)')
            # ax1.annotate(norm_amp, (df[column1].index[0], norm_amp), color='orange')
            ax1.annotate(amp, (df[column1].index[0], amp), color='blue')
        if norm_freq != None: 
            ax1.axhline(y=norm_freq, color='green', linestyle='-.', label=f'Average Usage Frequency (AAUF) in {self.groupby}')
            ax1.annotate(norm_freq, (df[column1].index[0], norm_freq), color='green')
        for i, segment in enumerate(features):
            for j, _segment in enumerate(segment[1]): 
                ax2.plot(_segment.index, _segment[column1])
                # ax2.annotate(f'{i}-{j}', (_segment.index[0], _segment[column1].max()))
        ax2.set_title(f'{column1} segments')
        ax1.set_xlim([df.index[0], df.index[-1]])
        ax2.set_xlim([df.index[0], df.index[-1]])
        ax2.set_ylim([0, df[column1].max()])
        ax2.set_ylabel('Power Consumption (W)')
        ax2.set_xlabel('Date')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        fig.tight_layout()
        fig.legend()
        plt.show()
####################
