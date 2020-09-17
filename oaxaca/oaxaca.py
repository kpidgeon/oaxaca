import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS


class Oaxaca:
    """
    Class that implements Oaxaca-Blinder decomposition for examining the effect
    of group differences on a continuous variable.
    """

    def __init__(self, data: pd.DataFrame, group_indicator: str, target: str):

        self.data = data
        self.group_indicator = group_indicator
        self.target = target

        if self.data[self.group_indicator].nunique() != 2:
            raise Exception(
                f"There must be exactly two groups, i.e. 2 unique values in column \
                {self.group_indicator}.")

    def _two_fold(self, _data: pd.DataFrame, coef_type: str, benchmark: int):
        """
        Parameters
        ----------
        coef_type : str
            The form of the reference coefficients for use in decomposition.
            This can be one of {benchmark, reimers, cotton, neumark, jann}
        """

        # need to fit a model for each group
        # need to decide on reference coefficients (coef_type dependent)

        groups = None
        if _data[self.group_indicator].nunique() != 2:
            raise Exception(
                f"There must be exactly two groups, i.e. 2 unique values in column \
                {self.group_indicator}.")
        else:
            groups = _data[self.group_indicator].unique()
            # print(groups)

        # TODO Ensure constant term in dataset

        # Here, we let group 'b' always be the non-discriminatory benchmark provided
        a_endo = _data[_data[self.group_indicator] != benchmark][self.target]
        b_endo = _data[_data[self.group_indicator] == benchmark][self.target]

        a_exog = _data[_data[self.group_indicator]
                       != benchmark].drop(columns=[self.target])
        b_exog = _data[_data[self.group_indicator]
                       == benchmark].drop(columns=[self.target])

        # print(groups)

        a_model = OLS(a_endo, a_exog)
        b_model = OLS(b_endo, b_exog)

        a_results = a_model.fit()
        b_results = b_model.fit()

        a_params = a_results.params
        b_params = b_results.params

        r_params = None
        if coef_type == 'benchmark':
            r_params = b_params
        elif coef_type == 'reimers':
            r_params = .5*a_params + .5*b_params
        elif coef_type == 'cotton':
            n_a_obs = len(a_endo)
            n_b_obs = len(b_endo)

            r_params = (1/(n_a_obs + n_b_obs)) * \
                (n_a_obs*a_params + n_b_obs*b_params)
        elif coef_type == 'neumark':
            # TODO Implement pooled regression models
            pass
        elif coef_type == 'jann':
            pass

        gap = a_endo.mean() - b_endo.mean()
        explained = (a_exog.mean() - b_exog.mean()).T * r_params
        a_unexplained = a_exog.mean().T * (a_params - r_params)
        b_unexplained = b_exog.mean().T * (r_params - b_params)

        return {'method': f'two_fold_{coef_type}',
                'outcome_gap': gap,
                'explained': explained,
                'a_unexplained': a_unexplained,
                'b_unexplained': b_unexplained,
                'a': (f'{self.group_indicator}', ~benchmark),
                'b': (f'{self.group_indicator}', benchmark)}

        #print(gap, explained.sum(), b_unexplained.sum())
        #print(explained.sum() + a_unexplained.sum() + b_unexplained.sum())

    def _three_fold(self):
        pass

    def decompose(self, method: str = 'two', benchmark: str = None, bootstrap: bool = False, bs_replicates: int = 100):

        if bootstrap:
            # call self._two_fold a total of 'bs_replicates' times, each with a bootstrapped sample
            # and store the results
            for _ in range(bs_replicates):
                _bootstrapped_data = self.data.sample(
                    n=len(self.data), replace=True)
                results = self._two_fold(
                    _data=_bootstrapped_data, coef_type='benchmark', benchmark=benchmark)
                # print(results['explained'].sum())
        else:
            if benchmark is None:
                pass
            else:
                results = self._two_fold(_data=self.data,
                                         coef_type='benchmark', benchmark=benchmark)

                # print(results)


class OaxacaResults:
    """
    For storing the results of a decomposition task.

    This will store the raw results (the components of the explained and
    unexplained portions of the discrepancy in group outcomes).
    It will also provide functionality to aggregate categorical dummies
    such that we see the overall contributions of categorical variables.
    """

    def __init__(self):
        pass
