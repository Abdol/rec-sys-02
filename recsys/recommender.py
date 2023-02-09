from enum import Enum
from matplotlib import patches, pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

verbose = False # TODO: use logging and argpass instead

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

# Class definition #
class RecommendationType(Enum):
    AMP = 1
    FREQ = 2
    OCC = 3
####################

# Class definition #
class Recommendation:
    def __init__(self, datetime, duration: int, app: Appliance, action: int, explanation: str, type: RecommendationType):
        self._datetime = datetime
        self._duration = duration
        self._app = app
        self._action = action
        self._explanation = explanation
        self._relevance = 0
        self.type = type

    @property
    def datetime(self):
        return self._datetime
    
    @property
    def duration(self):
        return self._duration

    @property
    def app(self):
        return self._app

    @property
    def action(self):
        return self._action

    @property
    def explanation(self):
        return self._explanation

    @property
    def relevance(self):
        return self.__calc_relevance()

    def __calc_normalization_factor(self):
        normalization_factor = 0
        # Convert self._app.groupby to int (secs)
        if self._app.groupby == '1min':
            normalization_factor = 60
        elif self._app.groupby == '1h':
            normalization_factor = 3600
        elif self._app.groupby == '1d':
            normalization_factor = 86400
        elif self._app.groupby == '1w':
            normalization_factor = 604800
        elif self._app.groupby == '1m':
            normalization_factor = 2592000
        return normalization_factor

    def __calc_relevance(self):
        """Calculate the relevance of the recommendation based on max power lost"""
        rel = 0
        duration = self._duration
        amp = self._app.norm_amp
        rel = int(duration * amp)
        return rel

    def __repr__(self) -> str:
        return f"Recommendation(datetime={self._datetime}, duration={self._duration}, app={self._app.column}, action={self._action}, explanation={self._explanation}, relevance={self._relevance})"
####################

# Class definition #
class Recommender:
    def __init__(self, app: Appliance, config: list[bool] = [True, True, True]):
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

    def savings(self, tariff):
        # TODO: fix calculation errors
        """
        Savings (WRONG):
        TV: 2.8562541666666665 £
        Toaster: 31.47722310908516 £
        Fridge: 0.28330513425183645 £
        Washing Machine: 36.945064384182054 £
        Computer 1: 0.00855507403451854 £
        Computer 2: 0.01115947418668977 £
        """
        """Calculate the total savings of the recommendations"""
        if len(self._recs) == 0:
            # Throw warning because recs is empty
            print('No recommendations generated yet. Call generate() first.')
            return
        recs = self.recs
        savings = 0
        for rec in recs:
            power_consumed = rec.app.norm_amp * rec.duration
            power_consumed /= 3600 # convert to kWh
            savings += power_consumed * tariff
        return savings

    def generate(self):
        if self.config[0]: self._recs += self.freq()
        if self.config[1]: self._recs += self.amp()
        if self.config[2]: self._recs += self.occ()

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

# Class definition #
class Evaluator:
    relevence_threshold = 0.5
    def __init__(self, rec: Recommender, y_pred = [], y_true = []):
        self._rec = rec
        self._y_true = y_true if len(y_true) != 0 else self.__estimate_y_true()
        self._y_pred = y_pred

    def __tp(self):
        y_true = self._y_true
        y_pred = self._y_pred
        if len(y_pred) == 0:
            return 0
        return sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])

    def __fp(self):
        y_true = self._y_true
        y_pred = self._y_pred
        if len(y_pred) == 0:
            return 0
        return sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])

    def __tn(self):
        y_true = self._y_true
        y_pred = self._y_pred
        if len(y_pred) == 0:
            return 0
        return sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0])

    def __fn(self):
        y_true = self._y_true
        y_pred = self._y_pred
        if len(y_pred) == 0:
            return 0
        return sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0])

    def __estimate_y_true(self):
        y_true = []
        relevence = self.relevance
        for _rel in relevence:
            if _rel > Evaluator.relevence_threshold:
                y_true.append(1)
            else:
                y_true.append(0)
        return y_true

    def __calc_relevance(self, recs=None):
        recs = self._rec.recs if recs is None else recs
        if len(recs) == 0:
            return [0]
        relevence = [rec.relevance for rec in recs]
        max_relevance = max(relevence)
        norm_relevence = [rel / max_relevance for rel in relevence]
        return norm_relevence

    def __calc_relevance_one(self, rel):
        relevence = [rec.relevance for rec in self._rec.recs]
        max_relevance = max(relevence)
        norm_relevence_one = rel / max_relevance
        return norm_relevence_one

    @property
    def relevance(self):
        return self.__calc_relevance()

    @property
    def rel_recs(self):
        rel = np.array(self.relevance)
        return [rec for rec in self._rec.recs if self.__calc_relevance_one(rec.relevance) > Evaluator.relevence_threshold]

    @property
    def precision(self):
        # rel_recs = len(self.rel_recs)
        # total_recs = len(self._rec.recs)
        # precision = rel_recs / total_recs
        tp = self.__tp()
        fp = self.__fp()
        if tp + fp == 0:
            return 0
        precision = tp / (tp + fp)
        return precision

    @property
    def recall(self):
        tp = self.__tp()
        fn = self.__fn()
        if tp + fn == 0:
            return 0
        recall = tp / (tp + fn)
        return recall

    @property
    def f1(self):
        if self.precision + self.recall == 0:
            return 0
        f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return f1

    @property
    def mae(self):
        n = len(self._rec.recs)
        if n == 0:
            return 0
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        mae = 1/n * sum(abs(y_true - y_pred))
        return mae

    @property
    def rmse(self):
        n = len(self._rec.recs)
        if n == 0:
            return 0
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        mse = 1/n * sum((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        return rmse

    @property
    def ap(self):
        n = len(self._rec.recs)
        if n == 0:
            return 0
        precision = np.array(self.precision)
        relevance = np.array(self.relevance)
        binary_relevance = np.array([1 if rel > Evaluator.relevence_threshold else 0 for rel in relevance])
        ap = 1/n * sum(precision * binary_relevance)
        return ap

    @property
    def ndcg(self):
        relevance = self.relevance
        n = len(self._rec.recs)
        if n == 0:
            return 0
        dcg = sum([relevance[i] / np.log2(i + 2) for i in range(n)])
        idcg = sum([1 / np.log2(i + 2) for i in range(n)])
        ndcg = dcg / idcg
        return ndcg

    @property
    def coverage(self):
        coverage = len(self._rec.recs) / len(self._rec.app.df)
        return coverage

    @property
    def diversity(self):
        # Compute similarity between recommendations
        # Create variable s that measure similarity between recommendations
        # The higher the value of s, the more similar the recommendations are
        # TODO: Write function that computes similarity between recommendations
        similarity = 0
        diversity = 1 - similarity
        return diversity

    @property
    def novelty(self):
        n = len(self._rec.recs)
        if n == 0:
            return 0
        total_relevance = sum([rec.relevance for rec in self._rec.recs])
        novelty = total_relevance / n
        return novelty

    def report(self):
        report = {
            'Precision': self.precision,
            'Recall': self.recall,
            'F1': self.f1,
            'MAE': self.mae,
            'RMSE': self.rmse,
            'AP': self.ap,
            'NDCG': self.ndcg,
            'Coverage': self.coverage,
            'Novelty': self.novelty,
            # 'Diversity (incomplete)': self.diversity,
        }
        return report
    
    def confusion_matrix(self):
        # Print confusion matrix as a nicely formatted table
        print("Confusion matrix:")
        print("y_true:\t   |\t1\t|\t0\t|")
        print("y_pred:\t 1 |\t{}\t|\t{}\t|".format(self.__tp(), self.__fp()))
        print("y_pred:\t 0 |\t{}\t|\t{}\t|".format(self.__fn(), self.__tn()))
###################

# Class defintion #
class Household:
    def __init__(self, recommenders: list[Recommender] = []):
        self.recommenders = recommenders
        self._recs = []
        self._evals = []

    def add_recommender(self, recommender):
        self.recommenders.append(recommender)

    def generate_recs(self):
        for recommender in self.recommenders:
            self._recs.append(recommender.generate())
            self._evals.append(Evaluator(recommender, recommender.y_pred))
        return self._recs

    def individial_report(self):
        for _eval in self._evals:
            print(_eval._rec._app.label, _eval.report())

    def evaluate(self, recs=None):
        # Compute average metrics
        avg_metrics = {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'MAE': 0.0,
            'RMSE': 0.0,
            'AP': 0.0,
            'NDCG': 0.0,
            'Coverage': 0.0,
            'Novelty': 0.0,
            # 'Diversity (incomplete)': self.diversity,
        }
        for _eval in self._evals:
            # Compute average precision from _eval in one line
            avg_metrics['Precision'] += _eval.precision
            avg_metrics['Recall'] += _eval.recall
            avg_metrics['F1'] += _eval.f1
            avg_metrics['MAE'] += _eval.mae
            avg_metrics['RMSE'] += _eval.rmse
            avg_metrics['AP'] += _eval.ap
            avg_metrics['NDCG'] += _eval.ndcg
            avg_metrics['Coverage'] += _eval.coverage
            avg_metrics['Novelty'] += _eval.novelty
        for key in avg_metrics:
            avg_metrics[key] /= len(self._evals)

        return avg_metrics   
####################