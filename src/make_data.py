import os
import gc
import sys
import warnings
warnings.filterwarnings('ignore')
from glob import glob
from copy import deepcopy
from collections import OrderedDict

import feather
import pandas as pd
import numpy as np

import jpholiday

from sklearn.preprocessing import LabelEncoder

# =============================================================================
# memory reducer
# =============================================================================
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:4] == 'date':
                pass
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# # =============================================================================
# # sales data
# # =============================================================================

# # load sales
# df = pd.read_csv('../input/sample/Sales.csv', encoding='CP932')
# df.columns = ['seg_code', 'date', 'sales_amt', 'num_regi_customer']

# # cleansing string to numerical
# df['sales_amt'] = df['sales_amt'].apply(lambda x: x.replace(',', '')).astype(int)
# df['num_regi_customer'] = df['num_regi_customer'].apply(lambda x: x.replace(',', '')).astype(int)
# df['seg_code'] = df['seg_code'].apply(lambda x: x.replace(',', ''))

# # date time preprocessing
# df['date']  = pd.to_datetime(df['date'], format='%Y/%m/%d')
# df['year']  = df['date'].dt.year
# df['month'] = df['date'].dt.month
# df['day']   = df['date'].dt.day
# df['dow']   = df['date'].dt.dayofweek
# df['week']  = df['date'].dt.week

# # extract holiday
# df['is_weekend'] = df['dow'].map(lambda x: 1 if x > 4 else 0)
# df['is_public_holiday'] = pd.Series(
#     pd.DatetimeIndex(df['date']).date
# ).map(jpholiday.is_holiday).astype(int)

# df['is_holiday'] = df['is_weekend'] + df['is_public_holiday']
# df['is_holiday'] = df['is_holiday'].map(lambda x: 1 if x > 0 else 0)

# # extract dummy holiday
# agg = df.groupby('date', as_index=False)[['sales_amt', 'is_holiday']].mean()
# agg['is_holiday'] = agg['is_holiday'].astype(int)

# dummy_holiday = agg[(agg['sales_amt'] > 200000) & (agg['is_holiday'] == 0)]
# dummy_holiday['dummy_holiday'] = 1

# df = df.merge(dummy_holiday[['date', 'dummy_holiday']], on='date', how='left')
# df['dummy_holiday'].fillna(0, inplace=True)
# df['dummy_holiday'] = df['dummy_holiday'].astype(int)

# del agg, dummy_holiday
# gc.collect()

# # extract month block (経過月数、経過日数の算出)
# df['month_block'] = pd.to_datetime(df['year'].astype('str') + '-' + df['month'].astype(str), format='%Y-%m')

# le = LabelEncoder()
# df['month_block'] = le.fit_transform(df['month_block'])
# df['month_block'] += 1


# # =============================================================================
# # shop data
# # =============================================================================

# # load shop
# shop = pd.read_csv('../input/sample/店舗マスタ_191004.csv', encoding='CP932', 
#                     dtype={'部署コード': 'object'})
# shop.columns = ['bland_code', 'bland_name', 'seg_code', 'shop_no', 'shop_name', 'pref_code', 'pref_name',
#                 'open_date', 'close_date', 'date', 'tenant_code', 'tenant_name']

# # date time
# shop['open_date'] = pd.to_datetime(shop['open_date'], format='%Y/%m/%d')
# shop['close_date'] = pd.to_datetime(shop['close_date'], format='%Y/%m/%d')

# # convert to string
# shop['seg_code'] = shop['seg_code'].astype(str)

# # merge data
# df = df.merge(shop.drop(columns=['bland_code', 'pref_code', 'date']), on='seg_code', how='left')

# # memory reduce
# df = reduce_mem_usage(df)

# # save file
# df.to_pickle(f'../input/pickle/preprocessed_data.pkl')
