# Imports #
from functions import *
from .recommender import Recommender
from .evaluator import Evaluator
####################

# Class defintion #
class Building:
    def __init__(self, recommenders: list[Recommender], tariff: float):
        self.recommenders = recommenders
        self._recs = []
        self._evals = []
        self._tariff = tariff

    def add_recommender(self, recommender):
        self.recommenders.append(recommender)

    def generate_recs(self):
        for recommender in self.recommenders:
            self._recs.append(recommender.generate())
            self._evals.append(Evaluator(recommender, tariff=self._tariff, y_pred=recommender.y_pred, y_true=[]))
        return self._recs

    def plot_recs(self):
        for rec in self.recommenders:
            rec.plot()

    def individial_report(self):
        report = []
        for _eval in self._evals:
            _report = (_eval._rec._app.label, _eval.report())
            report.append(_report)
        return report

    def report(self):
        # Compute average metrics
        evals = self._evals
        avg_metrics = evals[0].report()
        for _eval in evals:
            # Compute average precision from _eval in one line
            if _eval == evals[0]: continue
            avg_metrics['precision'] += _eval.precision
            avg_metrics['recall'] += _eval.recall
            avg_metrics['f1'] += _eval.f1
            avg_metrics['mae'] += _eval.mae
            avg_metrics['rmse'] += _eval.rmse
            avg_metrics['ap'] += _eval.ap
            avg_metrics['ndcg'] += _eval.ndcg
            avg_metrics['coverage'] += _eval.coverage
            avg_metrics['novelty'] += _eval.novelty
            avg_metrics['cost'] += _eval.savings
        for key in avg_metrics:
            avg_metrics[key] /= len(evals)

        return avg_metrics   
####################