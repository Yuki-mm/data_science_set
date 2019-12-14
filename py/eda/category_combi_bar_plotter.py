import pandas as pd
from typing import List
from eda import CategoricalTargetVsVarsBarPlotter
from feature import CategoryCombiFeatureCreator


class CategoryCombiBarPlotter:
    """
    """
    def __init__(self):
        self._creator = CategoryCombiFeatureCreator()
        self._plotter = CategoricalTargetVsVarsBarPlotter()

    def plot(self, df: pd.DataFrame, base_col: str, cols: List[str], group_col: str = None):
        """

        :param df:
        :param base_col:
        :param cols:
        :param group_col:
        :return:
        """
        _df = self._creator.create(df, base_col=base_col, cols=cols)
        _df[group_col] = df[group_col]
        self._plotter.plot(df=_df, tar_col=group_col)
