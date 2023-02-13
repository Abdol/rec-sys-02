# Imports #
from functions import *
from .recommender import Recommender
####################

# Class definition #
class Evaluator:
    relevence_threshold = 0.5
    def __init__(self, rec: Recommender, tariff: float, y_pred = [], y_true = []):
        self._rec = rec
        self._y_true = y_true if len(y_true) != 0 else self.__estimate_y_true()
        self._y_pred = y_pred
        self._tariff = tariff

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
        if len(self._y_pred) == 0:
            print('y_pred is empty')
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

    @property
    def savings(self):
        """Calculate the total savings of the recommendations"""
        if len(self._rec.recs) == 0:
            # Throw warning because recs is empty
            print('No recommendations generated yet. Call generate() first.')
            return
        recs = self._rec.recs
        tariff = self._tariff
        if tariff is None:
            print('No tariff specified.')
            return
        savings = 0.0
        for rec in recs:
            power_consumed = rec.app.norm_amp * rec.duration
            # print('power_consumed:', power_consumed, 'W', rec .duration, 's', rec.app.norm_amp, '(normal) W')
            power_consumed /= (3600 * 1000)  # convert to kWh
            # print('power_consumed:', power_consumed, 'kWh')
            savings += power_consumed * tariff
            savings /= 100  # convert to GBP
        return savings

    def report(self):
        report = {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'mae': self.mae,
            'rmse': self.rmse,
            'ap': self.ap,
            'ndcg': self.ndcg,
            'coverage': self.coverage,
            'novelty': self.novelty,
            'cost': self.savings
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
