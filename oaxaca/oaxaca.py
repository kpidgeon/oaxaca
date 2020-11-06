from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns

from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import RegressionResults

from typing import Iterable


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

    def _two_fold(self, _data: pd.DataFrame, coef_type: str, benchmark: int = 0):
        """
        Parameters
        ----------
        coef_type : str
            The form of the reference coefficients for use in decomposition.
            This can be one of {benchmark, reimers, cotton, neumark, jann}
        """

        # need to fit a model for each group
        # need to decide on reference coefficients (coef_type dependent)

        if _data[self.group_indicator].nunique() != 2:
            raise Exception(
                f"There must be exactly two groups, i.e. 2 unique values in column \
                {self.group_indicator}.")

        # TODO Ensure constant term in dataset

        # Here, we let group 'b' always be the non-discriminatory benchmark provided
        a_endo = _data[_data[self.group_indicator] != benchmark][self.target]
        b_endo = _data[_data[self.group_indicator] == benchmark][self.target]

        a_exog = _data[_data[self.group_indicator]
                       != benchmark].drop(columns=[self.target])
        b_exog = _data[_data[self.group_indicator]
                       == benchmark].drop(columns=[self.target])
        a_exog.drop(columns=[self.group_indicator], inplace=True)
        b_exog.drop(columns=[self.group_indicator], inplace=True)

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
                'a_reg_results': a_results,
                'b_reg_results': b_results,
                'a': (f'{self.group_indicator}', ~benchmark),
                'b': (f'{self.group_indicator}', benchmark)}

        #print(gap, explained.sum(), b_unexplained.sum())
        #print(explained.sum() + a_unexplained.sum() + b_unexplained.sum())

    def _three_fold(self):
        pass

    def decompose(self, method: str = 'two', benchmark: str = None,
                  bootstrap: bool = False, bs_replicates: int = 100):
        """
        Returns
        -------
        out : OaxacaResults or dict of OaxacaResults
            In the case of a single decomposition, a single OaxacaResults results
            container is returned.
            In the case of a bootstrapped decomposition, then a dictionary of all
            the individual OaxacaResults will be returned.
        """

        if bootstrap:
            # call self._two_fold a total of 'bs_replicates' times, each with a bootstrapped sample
            # and store the results

            # bootstrapped_results = {}

            bootstrapped_results = []

            for rep in range(bs_replicates):

                _bootstrapped_data = self.data.sample(
                    n=len(self.data), replace=True)
                    
                results = self._two_fold(
                    _data=_bootstrapped_data, coef_type='benchmark', benchmark=benchmark)

                bootstrapped_results.append(OaxacaResults(**results))

            return bootstrapped_results
            # OaxacaResults.plot_two_fold(results=bootstrapped_results, detailed=True)

        else:
            pass
            #     if benchmark is None:
            #         pass
            #     else:
            #         results = self._two_fold(_data=self.data,
            #                                  coef_type='benchmark', benchmark=benchmark)
            #         #

            #         return OaxacaResults(**results)


class OaxacaResults:
    """
    For storing the results of a decomposition task.

    This will store the raw results (the components of the explained and
    unexplained portions of the discrepancy in group outcomes).
    It will also provide functionality to aggregate categorical dummies
    such that we see the overall contributions of categorical variables.

    It will also provide functionality for determining confidence intervals
    from bootstrapped results?
    """

    def __init__(self, method: str, outcome_gap: float,
                 a_reg_results: RegressionResults,
                 b_reg_results: RegressionResults,
                 **kwargs):
        self.method = method
        self.outcome_gap = outcome_gap
        self.a_reg_results = a_reg_results
        self.b_reg_results = b_reg_results

        for k in kwargs:
            setattr(self, k, kwargs[k])

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def get_twofold_info(results: Iterable[OaxacaResults],
                      detailed: bool = False,
                      component: str = 'explained',
                      ci=95):
        """

        """
        if detailed:
            # get all bootstrapped data for given component
            data = [getattr(res, component) for res in results]
            concat_data = pd.concat(data)
            avg_data = concat_data.groupby(concat_data.index).mean()
            errors = concat_data.groupby(concat_data.index).apply(
                lambda x: np.percentile(x, q=[(100-ci)/2, 100-(100-ci)/2]))

            return pd.DataFrame({component: avg_data, 'error': errors})

        else:
            pass

    # @staticmethod
    # def calc_ci(results: Iterable[OaxacaResults],
    #             component: str):
    #     """
    #     Calculate confidence intervals for a decomposition component.

    #     """

    #     values = [getattr(res, component)
    #               for res in results if hasattr(res, component)]

    #     return np.percentile()
