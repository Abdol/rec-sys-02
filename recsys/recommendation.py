# Class definition #
# Imports #
from functions import *
from .appliance import Appliance
####################

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
