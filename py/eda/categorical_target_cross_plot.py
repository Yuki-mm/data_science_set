import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from matplotlib.figure import Figure


class CategoricalTargetCrossPlot:
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
    COL_NAME_CR = 'COUNT_RATE'
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

    def __init__(self):
        self._tar_rate_basis = None
        self._text_kwargs = dict(fontsize=self.TEXT_FONT_SIZE)

    def plot(self, df: pd.DataFrame, tar_col: str):
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
            self._flow_category(df=df, col=col, tar_col=tar_col)

        # 目的変数vs量的変数の描画
        num_cols = df.select_dtypes(include='number').columns
        for col in num_cols:
            self._flow_numerical(df=df, col=col, tar_col=tar_col)

    def _flow_category(self, df, col, tar_col):
        """ カテゴリ変数の処理フロー

        :param df: col、tar_colを含むDataFrame
        :param col: 目的変数比較対象のカテゴリ型のカラム名
        :param tar_col: 目的変数のカラム名
        :return: なし
        """
        # TODO: 一旦後回し
        return

    def _flow_numerical(self, df, col, tar_col):
        """ 量的変数の処理フロー

        :param df: col、tar_colを含むDataFrame
        :param col: 目的変数比較対象の数値型のカラム名
        :param tar_col: 目的変数のカラム名
        :return: なし
        """
        if col == tar_col:
            return

        # カテゴリ数が多いものはグループ化して分析する
        _df = df[[col, tar_col]]
        _value_counts_df = self._cvt_targ_rate_df(df=df, col=col, tar_col=tar_col)
        is_val, msgs = self._validation_category(df=_value_counts_df, col=col)
        is_num_group = False
        if not is_val:
            s1 = pd.cut(_df[col], 10)
            _df[col] = s1
            is_num_group = True

        _cross_df = pd.crosstab(_df[tar_col], columns=[_df[col]])
        _cross_df = _cross_df.apply(lambda x: x / sum(x), axis=1)

        # カテゴリ値単位の目的変数１割合折れ線グラフとカテゴリ値カウントの棒グラフを描画
        fig, ax = self._get_fig_obj(df=_value_counts_df, is_num_group=is_num_group)
        self._plot_target_rate(df=_cross_df, ax=ax[0])

        _cross_df = pd.crosstab(_df[tar_col], columns=[_df[col]])
        self._plot_target_rate(df=_cross_df, ax=ax[1], fmt='1')

        # 出力
        # self._show(ax=ax, title=f'Numerical Variable({col}) vs Target')

    def _show(self, ax, title: str):
        """ 描画オブジェクトの出力処理

        :param ax: 描画オブジェクト
        :param title: グラフの上に出力するタイトル
        :return: なし
        """
        ax.set_title(title, **self._text_kwargs)
        plt.show()

    def _get_fig_obj(self, df: pd.DataFrame, is_num_group: bool = False):
        """ 描画オブジェクト取得

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :param is_num_group: 数値型変数をグループ化していた場合はTrue、それ以外はFalse（デフォルトはFalse）
        :return: Figureオブジェクト, AxesSubplotオブジェクト
        """
        x_fs, y_fs = self._get_figsize(df=df)

        # 数値型のグループ化描画時はxラベルが長くなるため、強制的に横長にする
        if is_num_group:
            x_fs = self.FIGSIZE_X_LEARGE

        fig, ax = plt.subplots(1, 2,
                               figsize=(x_fs, y_fs),
                               linewidth=self.FIG_LINE_WIDTH,
                               edgecolor=self.FIG_LINE_COLOR)

        return fig, ax

    def _cvt_targ_rate_df(self, df: pd.DataFrame, col: str, tar_col: str) -> pd.DataFrame:
        """ 対象カラムの値ごとに目的変数が１である割合を算出

        :param df: col、tar_colを含むDataFrame
        :param col: 対象カラム名
        :param tar_col: 目的変数のカラム名
        :return: 対象カラムの値ごとに目的変数が１である割合の算出結果DataFrame
        """
        dct = {
            self.COL_NAME_CNT: df[col].value_counts(),
            self.COL_NAME_CR: df[col].value_counts(normalize=True),
            self.COL_NAME_TR: df.groupby(col).agg({tar_col: 'mean'})[tar_col]
        }

        df = pd.DataFrame(dct)

        # 棒グラフ表示は左から大きい値にしたいため並び替え
        df.sort_values(self.COL_NAME_CNT, ascending=False, inplace=True)
        return df

    def _cvt_num_group_df(self, df: pd.DataFrame, col: str, tar_col: str) -> pd.DataFrame:
        """ 数値型の値をグループ化

        数値型で取りうる値の範囲が多すぎる場合にもざっくりと傾向を掴みたいため
        グループ化した結果を返す

        :param df: col、tar_colを含むDataFrame
        :param col: 対象カラム名
        :param tar_col: 目的変数のカラム名
        :return: 対象数値型カラムの値をグループ化し、目的変数が１である割合の算出結果DataFrame
        """

        # 値の取りうる範囲を指定の値で分割
        df_cut = pd.cut(df[col], self.GROUP_CUT_NUM).to_frame()

        # 特定の範囲で数値をグループ化する、型がintervalになるため扱いやすいように文字列変換
        df_cut = df_cut.astype('str')

        # 元のカラムは消し、新しいカラムを元のカラム名でマージ
        df = pd.concat([df_cut, df.drop(col, axis=1)], sort=False, axis=1)

        return self._cvt_targ_rate_df(df=df, col=col, tar_col=tar_col)

    def _validation_category(self, df: pd.DataFrame, col: str) -> (bool, [str]):
        """ カテゴリ値のチェック

        チェックルールは以下の通り
        ・変数の取りうる値が多すぎる場合はエラー（描画不可）

        :param df: col、tar_colを含むDataFrame
        :param col: チェック対象カラム名
        :return: チェック結果（True:OK／False:NG）, エラーメッセージリスト（エラーなし時は空配列）
        """
        msgs = []

        # カテゴリ数が多いものは描画しきれないため対象外とする
        cate_cnt = len(df[self.COL_NAME_CNT])
        if cate_cnt > self.PLOT_CATEGORY_MAX:
            msgs.append(f'{col} is many category! (num={cate_cnt})')
            return False, msgs

        return True, []

    def _get_figsize(self, df: pd.DataFrame) -> (int, int):
        """ 描画オブジェクトのサイズ取得

        カテゴリ数が多いものは文字が重ならないようにx軸を大きめにとる

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :return: 横のサイズ、縦のサイズ
        """
        x_fs = self.FIGSIZE_X_DEFAULT
        y_fs = self.FIGSIZE_Y_DEFAULT
        cate_cnt = len(df[self.COL_NAME_CNT])
        if cate_cnt > self.FIGSIZE_X_SIZE_TH:
            x_fs = self.FIGSIZE_X_LEARGE

        return x_fs, y_fs

    def _get_ylim(self, df: pd.DataFrame) -> float:
        """ Y軸のメモリMAXサイズ取得

        目的変数比率が0.5以上のものがあればメモリは1.0、違う場合は0.5とする（細かくは設定しない）

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :return: Y軸のメモリサイズ
        """
        y_lim = self.PLOT_Y_LIM_DEFAULT
        tr_max = df.TARGET_RATE.max()
        if tr_max > self.PLOT_Y_LIM_DEFAULT:
            y_lim = self.PLOT_Y_LIM_MAX

        return y_lim

    @staticmethod
    def _get_x_labels(df: pd.DataFrame) -> np.ndarray:
        """ x軸表示リスト取得

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :return: x軸の表示リスト
        """
        return df.index.values

    def _plot_target_rate(self, df: pd.DataFrame, ax, fmt='1.2f'):
        """

        :param df: _cvt_targ_rate_df で生成したDataFrame
        :param ax: 描画オブジェクト
        :return: なし
        """
        # 目的変数vsカテゴリ変数のカウントを棒グラフで描画
        sns.heatmap(df, annot=True, fmt=fmt, cmap=self.MAP_COLOR, ax=ax)
