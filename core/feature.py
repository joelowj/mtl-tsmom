#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import numpy as np
import pandas as pd


from functools import reduce
from typing import List


"""
    For the feature generation code below, `df` columns are 'date', 'ticker', 'close'.
    The feature generation here are kept as simple as possible, please read paper for reasoning.
    If you have more predictive features, feel free to use them as inputs.
"""


def calc_holding_period_ret(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    for period in periods:
        df[f'ret_{period}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(period))
    return df


def calc_ex_ante_vol(df: pd.DataFrame, spans: List[int], ann_factor: int = 252) -> pd.DataFrame:
    drop_cols = []
    if 'ret_1d' not in df.columns:
        df['ret_1d'] = df.groupby('ticker')['close'].transform('pct_change', periods=1, fill_method=None)
        drop_cols.append('ret_1d')
    df['ret_1d_sq'] = df['ret_1d'] ** 2
    drop_cols.append('ret_1d_sq')
    for span in spans:
        col_id = f'ex_ante_vol_{span}'
        df[col_id] = df.groupby('ticker')['ret_1d_sq'].transform(lambda x: x.ewm(span=span).mean())
        df[col_id] = np.sqrt(df[col_id]) * np.sqrt(ann_factor)
    df.drop(columns=drop_cols, inplace=True)
    return df


def calc_mom_ret_feature(df: pd.DataFrame, periods: List[int], clip_val: int = 2) -> pd.DataFrame:
    for period in periods:
        vol_category = 'fast' if period in [1, 3, 5, 10] else ('mid' if period in [21, 63] else 'slow')
        df[f'feature_mom_ret_{period}'] = df[f'ret_{period}'] / df[f'ex_ante_vol_{vol_category}']
        df[f'feature_mom_ret_{period}'] = df[f'feature_mom_ret_{period}'].clip(lower=-clip_val, upper=clip_val)
    return df


def generate_tsmom_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    periods = [1, 3, 5, 10, 21, 63, 126, 252]
    df = calc_holding_period_ret(df, periods=periods)
    df = calc_ex_ante_vol(df, spans=[3, 5, 10, 21, 63, 126, 252])
    df['ex_ante_vol_fast'] = df[[f'ex_ante_vol_{span}' for span in [3, 5, 10]]].max(axis=1)
    df['ex_ante_vol_mid'] = df[[f'ex_ante_vol_{span}' for span in [21, 63]]].max(axis=1)
    df['ex_ante_vol_slow'] = df[[f'ex_ante_vol_{span}' for span in [126, 252]]].max(axis=1)
    df = calc_mom_ret_feature(df, periods)
    feature_cols = [col for col in df.columns if 'feature' in col]
    df = df[['date', 'ticker'] + feature_cols]
    df.loc[:, feature_cols] = df[feature_cols].dropna(axis=0, how='all')
    df = df.fillna(0.)
    return df.set_index(['date', 'ticker'])