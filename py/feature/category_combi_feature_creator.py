class CategoryCombiFeatureCreator:
    SEP = '-'

    def __init__(self):
        pass

    def create(self, df, base_col, cols):
        filter_cols = cols.copy()
        filter_cols.append(base_col)
        _df = df.loc[:, filter_cols]

        # 離散値型が来た場合を考慮し文字列型へ変換
        _df = _df.applymap(str)

        s1 = _df[base_col]
        for x in cols:
            s2 = _df[x]
            combi_col = base_col + self.SEP + x
            _df[combi_col] = s1 + self.SEP + s2
            _df.drop(x, axis=1, inplace=True)

        _df.drop(base_col, axis=1, inplace=True)
        return _df
