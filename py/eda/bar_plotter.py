from typing import List

import pandas as pd
import seaborn as sns

from const import BarPlotDefaultConst as DefConst
from const import OrientConst
from eda import BasePlotter


class BarPlotter(BasePlotter):
    def __init__(self):
        pass

    def plot(self,
             ax,
             df: pd.DataFrame,
             col: str,
             order: List[str] = None,
             group_col: str = None,
             group_col_order: List[str] = None,
             is_norm: bool = False,
             title: str = None,
             is_horizon: bool = False):

        sns_params = {
            'ax': ax,
            'data': df,
            'x': col,
            'order': order,
            'hue': group_col,
            'hue_order': group_col_order,
            'palette': DefConst.PALETTE,
            'dodge': True,
            'alpha': DefConst.ALPHA,
            'orient': DefConst.ORIENT
        }

        if is_horizon:
            sns_params['orient'] = OrientConst.HORIZON

        if is_norm:
            # 0〜1への正規化はオプション指定などないので estimator で変換
            sns.barplot(y=col,
                        estimator=lambda x: len(x) / len(df),
                        **sns_params)

            if is_horizon:
                ax.set_xlim([0, 1])
                ax.set_xlabel(DefConst.RATIO_LABEL)
            else:
                ax.set_ylim([0, 1])
                ax.set_ylabel(DefConst.RATIO_LABEL)

        else:
            # sns.countplot(**sns_params)
            sns.barplot(y=col,
                        estimator=lambda x: len(x),
                        **sns_params)

        # タイトル指定があれば表示
        if title:
            ax.set_title(title)
