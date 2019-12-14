import pandas as pd
from IPython.display import display as ipy_display


class TableOverview:
    """ テーブル概要取得

    Example:
        tov = TableOverview(df_train)
        print(f'rows={tov.row_cnt} / cols={tov.col_cnt}')
        tov.df_table_info

    Note:
    """
    def __init__(self, df: pd.DataFrame):
        self._row_cnt = df.shape[0]
        self._col_cnt = df.shape[1]
        self._df_table_info = df.dtypes.to_frame().rename(columns={0: 'TYPE'})
        self._df_table_info['NULL_CNT'] = df.isnull().sum()
        self._df_table_info['NULL_RATE'] = self._df_table_info.NULL_CNT / self._row_cnt
        self._df_table_info = pd.concat([self._df_table_info, df.describe(include='all').T],
                                        axis=1,
                                        sort=False)

        # カラム名は大文字に変換（カラムを区別しやすくするため大文字としておく）
        self._df_table_info.columns = [x.upper() for x in self._df_table_info.columns]

        # カラムの順番指定（データ内に数値型があれば追加）
        columns_sort = ['TYPE', 'COUNT', 'NULL_CNT', 'NULL_RATE']
        if len(df.select_dtypes(include='number').columns) > 0:
            columns_sort += ['MEAN', 'STD', 'MIN', '25%', '50%', '75%', 'MAX']

        # カラムの順番指定（データ内にテキスト型があれば追加）
        if len(df.select_dtypes(include='object').columns) > 0:
            columns_sort += ['UNIQUE', 'TOP', 'FREQ']

        self._df_table_info = self._df_table_info[columns_sort]

    def disp_table_info(self):
        ipy_display(self._df_table_info)

    def disp_data_volume(self):
        data = {
            'COL_CNT': [self._col_cnt],
            'ROW_CNT': [self._row_cnt]
        }
        _df = pd.DataFrame(data)
        ipy_display(_df)
