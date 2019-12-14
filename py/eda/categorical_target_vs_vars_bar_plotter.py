import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display as ipy_display
import japanize_matplotlib
from matplotlib.figure import Figure


class CategoricalTargetVsVarsBarPlotter:
    """
    各カラムの目的変数が「１」である割合をグラフ描画
    """
    FIGSIZE_X_DEFAULT = 16
    FIGSIZE_Y_DEFAULT = 4
    FIGSIZE_X_LEARGE = 16
    FIGSIZE_X_SIZE_TH = 8  # 描画パレットを大きくするかの閾値
    PLOT_CATEGORY_MAX = 8
    PLOT_Y_LIM_DEFAULT = 0.5
    PLOT_Y_LIM_MAX = 1.0
    COL_NAME_CNT = 'COUNT'
    COL_NAME_CR = 'COUNT_RATIO'
    COL_NAME_TR = 'TARGET_RATE'
    GROUP_CUT_NUM = 8

    PLOT_LINESTYLE_SOLID = 'solid'
    PLOT_LINESTYLE_DASHED = 'dashed'
    PLOT_LINESTYLE_DASHDOT = 'dashdot'
    PLOT_LINESTYLE_DOTTED = 'dotted'
    TEXT_FONT_SIZE = 12

    PLOT_COLOR_RED = 'Reds'
    PLOT_COLOR_BLUE = 'b'
    PLOT_COLOR_GRAY = 'gray'

    TR_LINE_COLOR = PLOT_COLOR_BLUE
    TR_LINE_WIDTH = 2
    MAP_COLOR = PLOT_COLOR_RED

    FIG_LINE_WIDTH = 5
    FIG_LINE_COLOR = PLOT_COLOR_GRAY

    BAR_Y_DATA_ADJUST = 0.8  # 棒グラフがグラフの一番上まで引かれることがないよう調整（見た目のお話）
    BAR_Y_DATA_COLOR = PLOT_COLOR_GRAY
    BAR_Y_DATA_ALPHA = 0.5  # 透明度を0～1で指定
    BAR_Y_DATA_TEXT_POSITION = 'center'

    MODE_COUNT = '1'
    MODE_RATIO = '2'

    def __init__(self, mode=MODE_COUNT):
        self._tar_rate_basis = None
        self._text_kwargs = dict(fontsize=self.TEXT_FONT_SIZE)
        self._mode = mode

    def plot(self, df: pd.DataFrame, tar_col: str = None):
        """ グラフ描画処理

        以下のルールで目的変数と各要素の関連のグラフを描画する
        ・各要素のカテゴリ値において目的となる変数が１の割合を折れ線グラフで描画（青の点線）
        ・各要素のカテゴリ値の件数を棒グラフで描画

        件数は多いが、目的変数１の割合が低い　　→　負の相関の可能性あり
        件数は少ないが、目的変数１の割合が多い　→　正の相関の可能性あり

        目的変数１の割合が多い、少ないの基準は赤の点線でひいた補助線で判断

        :param df: グラフ可視化対象のDataFrame、目的変数を含んでいること
        :param tar_col: 目的変数（0 or 1の値であること）
        :return: なし
        """

        # 目的変数vsカテゴリ変数の描画
        obj_cols = df.select_dtypes(include='object').columns
        for col in obj_cols:
            _df = df[[tar_col, col]]
            # カテゴリ数が多いものは描画しきれないため対象外としSkipする
            try:
                self._validation_category(df=_df, col=col)
            except Exception as e:
                print(e)
                continue

            x_labels = self._get_x_labels(df=_df, col=col)
            ylim = self._get_ylim(df=_df, col=col)     # Y軸に表示するスケール値のMAX
            fig, axs = self._get_fig_obj()

            # 目的変数別に可視化
            for ax, (_df2, tar_val) in zip(axs, self._target_dataset_split(df=df, tar_col=tar_col)):
                self._flow_category(df=_df2, col=col, tar_col=tar_col, ax=ax, x_labels=x_labels)
                ax.set_ylim(ylim)
                ax.set_title(f'target = {str(tar_val)}', **self._text_kwargs)

            plt.tight_layout()                      # 重なって表示されることを防ぐ
            fig.suptitle(col, **self._text_kwargs)  # Figureオブジェクトのタイトル表示
            plt.subplots_adjust(top=0.85)           # Figureオブジェクトのタイトルが被らないよう調整
            plt.show()

    @staticmethod
    def _target_dataset_split(df, tar_col):
        count_df = df[tar_col].value_counts()
        for x in count_df.index:
            yield (df[df[tar_col] == x], x)

    def _flow_category(self, df, col, tar_col, ax, x_labels):
        """ カテゴリ変数の処理フロー

        :param df: col、tar_colを含むDataFrame
        :param col: 目的変数比較対象のカテゴリ型のカラム名
        :param tar_col: 目的変数のカラム名
        :return: なし
        """
        # 目的変数が「1」である割合のデータセット生成
        # グローバル領域の変数と被らないように頭にアンダースコアをつけておく
        _df = self._cvt_targ_rate_df(df=df, col=col, tar_col=tar_col, x_labels=x_labels)

        # カテゴリ値単位の目的変数１割合折れ線グラフとカテゴリ値カウントの棒グラフを描画
        self._plot_count(df=_df, ax=ax, x_labels=x_labels)

    def _get_fig_obj(self):
        """ 描画オブジェクト取得

        :return: Figureオブジェクト, AxesSubplotオブジェクト
        """
        x_fs = self.FIGSIZE_X_DEFAULT
        y_fs = self.FIGSIZE_Y_DEFAULT

        fig, ax = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(x_fs, y_fs),
                               linewidth=self.FIG_LINE_WIDTH,
                               edgecolor=self.FIG_LINE_COLOR)

        return fig, ax

    def _cvt_targ_rate_df(self, df: pd.DataFrame, col: str, tar_col: str, x_labels) -> pd.DataFrame:
        """ 対象カラムの値ごとに目的変数が１である割合を算出

        :param df: col、tar_colを含むDataFrame
        :param col: 対象カラム名
        :param tar_col: 目的変数のカラム名
        :return: 対象カラムの値ごとに目的変数が１である割合の算出結果DataFrame
        """
        data = {
            self.COL_NAME_CNT: df[col].value_counts(),
            self.COL_NAME_CR: df[col].value_counts(normalize=True)
        }

        _df = pd.DataFrame(data)

        # 0件であってもラベルは表示したいため、0件データを追加
        if len(_df) != len(x_labels):
            data2 = {
                self.COL_NAME_CNT: [0 for x in x_labels],
                self.COL_NAME_CR: [0.0 for x in x_labels]
            }
            _df2 = pd.DataFrame(data2, index=x_labels)
            _df3 = pd.concat([_df, _df2])
            _df = _df3.sum(level=0)

        # 棒グラフ表示は左から大きい値にしたいため並び替え
        _df.sort_values(self.COL_NAME_CNT, ascending=False, inplace=True)
        return _df

    def _validation_category(self, df: pd.DataFrame, col: str) -> (bool, [str]):
        """ カテゴリ値のチェック

        チェックルールは以下の通り
        ・変数の取りうる値が多すぎる場合はエラー（描画不可）

        :param df: col、tar_colを含むDataFrame
        :param col: チェック対象カラム名
        :return: チェック結果（True:OK／False:NG）, エラーメッセージリスト（エラーなし時は空配列）
        """
        # カテゴリ数が多いものは描画しきれないため対象外とする
        cate_cnt = len(df[col].unique())
        if cate_cnt > self.PLOT_CATEGORY_MAX:
            raise ValueError(f'{col} is many category! (num={cate_cnt})')

    @staticmethod
    def _get_x_labels(df: pd.DataFrame, col) -> np.ndarray:
        """ x軸表示リスト取得

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :return: x軸の表示リスト
        """
        _df = df[col].value_counts()
        _df.sort_values(0, ascending=False, inplace=True)
        return _df.index.values

    def _get_ylim(self, df: pd.DataFrame, col) -> np.ndarray:
        """ x軸表示リスト取得

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :return: x軸の表示リスト
        """
        if self._mode == self.MODE_COUNT:
            max_val = df[col].value_counts()[0]  # value_countsの先頭が1番件数の多い要素

            # MAX値の少し上でかつキリのいい数字が上限になるよう設定
            tmp = (10 ** (len(str(int(max_val / 1)))) / 10)
            y_max = int(max_val / tmp + 1) * tmp
            return [0, y_max]
        elif self._mode == self.MODE_RATIO:
            return [0, 1.0]

    def _plot_count(self, df: pd.DataFrame, ax, x_labels):
        """

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :param ax: 描画オブジェクト
        :return: なし
        """
        if self._mode == self.MODE_COUNT:
            y_datas = df.loc[x_labels, self.COL_NAME_CNT]
            y_label = self.COL_NAME_CNT
        elif self._mode == self.MODE_RATIO:
            y_datas = df.loc[x_labels, self.COL_NAME_CR]
            y_label = self.COL_NAME_CR

        ax.bar(x=x_labels,
               height=y_datas,
               facecolor=self.BAR_Y_DATA_COLOR,
               alpha=self.BAR_Y_DATA_ALPHA)

        ax.set_ylabel(y_label, **self._text_kwargs)

        # データの上に件数を表示（Y軸のメモリとは連動させないため）
        y_counts = df.loc[x_labels, self.COL_NAME_CNT]
        for x, y, c in zip(x_labels, y_datas, y_counts):
            ax.text(x, y, str(c), ha=self.BAR_Y_DATA_TEXT_POSITION)
