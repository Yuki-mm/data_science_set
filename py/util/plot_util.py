from matplotlib import pyplot as plt
from const import FigureConst as fig_const


def get_fig_obj(row_cnt: int, col_cnt: int, row_size: int = None, col_size: int = None):
    """ 描画オブジェクト取得

    :return: Figureオブジェクト, AxesSubplotオブジェクト
    """
    col_size = col_size if col_size is not None else fig_const.COL_DEFAULT_SIZE
    row_size = row_size if row_size is not None else fig_const.ROW_DEFAULT_SIZE

    fig, ax = plt.subplots(nrows=row_cnt,
                           ncols=col_cnt,
                           figsize=(col_size, row_size),
                           linewidth=fig_const.LINE_WIDTH,
                           edgecolor=fig_const.LINE_COLOR)

    return fig, ax
