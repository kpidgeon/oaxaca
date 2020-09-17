import pandas as pd
import numpy as np




class Oaxaca:
    """
    Class that implements Oaxaca-Blinder decomposition for examining the effect
    of group differences on a continuous variable.
    """

    def __init__(self, data: pd.DataFrame, group_indicator: str, target: str):
        pass

    def _two_fold(self, coef_type: str = 'benchmark'):
        """
        Parameters
        ----------
        coef_type : str
            The form of the reference coefficients for use in decomposition.
            This can be one of {benchmark, reimers, cotton, neumark, jann}
        """
        pass

    def _three_fold(self):
        pass

    def decompose(self, method: str = 'two', benchmark: str = None, **kwargs):
        
        if benchmark is None:

