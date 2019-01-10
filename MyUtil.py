#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import time
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import requests
import argparse
from io import StringIO
from constants import (REGION, BJ_STATION_CHN_PINGYING_MAPPING, LONDON_API_COL_MAPPING,
                       AQ_COL_NAMES, DATA_API_TOKEN, SUB_TOKEN, USERNAME, MONTH_DIGIT_STR_MAPPING
                       )
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
import pickle
import datetime
import constants
from datetime import date, timedelta

class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        self.mean = np.nanmean(values)
        self.std = np.nanstd(values)
        return self

    def transform(self, values):
        return (np.array(values)-self.mean) / self.std

    def inverse_transform(self, values):
        return self.mean + self.std * np.array(values)

    def fit_transform(self, values):
        self.mean = np.nanmean(values)
        self.std = np.nanstd(values)
        return self.transform(values)

    def mean_(self):
        assert self.mean is not None, 'not fitted yet'
        return self.mean

    def std_(self):
        assert self.std is not None, 'not fitted yet'
        return self.std

    def __repr__(self):
        return 'scalar with mean {:10.4f} and std {:10.4f}'.format(self.mean, self.std)


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    https://github.com/facebook/prophet/issues/223
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def str2bool(v):
    """
    used to parse string argument true and false to bool in argparse
    # https://stackoverflow.com/questions/15008758/
    # parsing-boolean-values-with-argparse
    """
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


@contextmanager
def timer(description):
    """ time the run time of the chunk of code """
    t0 = time.time()
    yield
    print(f'[{description}] done in {time.time() - t0:.0f} s')


def get_verbose_print(verbose_level):
    if verbose_level > 0:
        def vprint(*vargs):
            if verbose_level >= vargs[0]:
                print(vargs[1])
    else:
        def vprint(*vargs):
            pass
    setattr(vprint, 'verbose_level', verbose_level)
    return vprint


def get_date(datetime_obj):
    return str(datetime_obj).split()[0]


def lgbm_impute(data: pd.DataFrame, vprint=print) -> pd.DataFrame:
    assert os.path.exists('models/sing_lgbm.pkl'), 'lgb model not trained yet'

    vprint(1, 'impute data with lgbm')
    with suppress_stdout_stderr():
        lgb_models = pickle.load(open('models/sing_lgbm.pkl', 'rb'))
    measure, regions, threshold = 'Value', REGION, 2
    dfs = {}
    x_data = [data.loc[data.Region == region, measure].values for region in regions]
    x_data = pd.DataFrame(np.array(x_data)).T
    x_data.columns = regions
    dfs[measure] = x_data.copy()
    for region in regions:
        vprint(2, "impute {} - {} with lgb".format(region, measure))
        value = dfs[measure][region].copy()
        condition = [not i for i in value.isnull()] & (dfs[measure].isnull().sum(axis=1) < threshold)
        predicted_temp = lgb_models['{}-{}'.format(region, measure)]
        cols = []
        for col in regions:
            if col != region:
                cols.append(col)
        dfs_temp = dfs[measure].loc[condition, cols]
        print(dfs_temp)
        predicted = predicted_temp.predict(dfs_temp)
        value[condition] = predicted
        data.loc[data['Region'] == region, measure] = value.tolist()
    return data


def forward_backward_impute(data: pd.DataFrame, method: str) -> pd.DataFrame:
    assert method in ['day', 'hour', 'fwbw_day', 'fwbw_hour'], 'method of grouping not correct'
    df = data.copy()  # type: pd.DataFrame
    if method == 'day' or method == 'fwbw_day':
        df.loc[:, 'key'] = df['utc_time'].map(get_date)
    if method == 'hour' or method == 'fwbw_hour':
        df.loc[:, 'key'] = df['utc_time'].map(lambda x: str(x.hour))
    df.sort_values(by=['stationId', 'key'], axis=0, inplace=True)
    df = df.groupby(['stationId', 'key'], sort=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    df.drop(labels='key', axis=1, inplace=True)
    df.sort_values(by=['stationId', 'utc_time'], axis=0, inplace=True)
    return df


def mean_impute(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()  # type: pd.DataFrame
    df = df.groupby('stationId', sort=False).apply(lambda x: x.fillna(x.mean())).reset_index(drop=True)
    return df


def impute(data: pd.DataFrame, lgbm: bool=False, day: bool=False, hour: bool=False, mean: bool=False, zero: bool=False,
           vprint=print) -> pd.DataFrame:
    df = data.copy()  # type: pd.DataFrame
    if lgbm:
        df.loc[df['stationId'].isin(BEIJING_STATIONS)] = lgbm_impute(
            data=df.loc[df['stationId'].isin(BEIJING_STATIONS)], city='bj', vprint=vprint)
        df.loc[df['stationId'].isin(LONDON_STATIONS)] = lgbm_impute(
            data=df.loc[df['stationId'].isin(LONDON_STATIONS)], city='ld', vprint=vprint)
    if day:
        vprint(2, 'day impute')
        df = forward_backward_impute(data=df, method='day')
    if hour:
        vprint(2, 'hour impute')
        df = forward_backward_impute(data=df, method='hour')
    if mean:
        vprint(2, 'mean impute')
        df = mean_impute(data=df)
    if zero:
        vprint(2, 'zero impute')
        df.fillna(0, inplace=True)
    return df


def fwbwmean_impute(data: pd.DataFrame, main_method: str='mean', residual_method: str='mean'):
    # TODO: use above one instead
    df = data.copy()  # type: pd.DataFrame
    if main_method == 'fwbw_day' or main_method == 'day':
        df = impute(df, day=True)
    if main_method == 'fwbw_hour' or main_method == 'hour':
        df = impute(df, hour=True)
    if main_method == 'mean' or residual_method == 'mean':
        df = impute(df, mean=True)
    if residual_method == 'zero':
        df = impute(df, zero=True)
    return df


def smape(y_true, y_pred):
    """ https://biendata.com/competition/kdd_2018/evaluation/ """
    denom = (y_true + y_pred) / 2
    ape = np.abs(y_true - y_pred) / denom
    ape[denom == 0] = 0
    return np.nanmean(ape)


def official_smape(actual, predicted):
    """ https://biendata.com/competition/kdd_2018/evaluation/ """
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))


def fix_nat(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()  # type: pd.DataFrame
    df['Value'] = df['Value'].map(lambda x: x if str(x) != 'NaT' else np.nan)
    return df


def long_to_wide(data, index='Region', columns='utc_time'):
    df = data.copy()  # type: pd.DataFrame
    data = df.pivot(index=index, columns=columns, values='Value').reset_index()
    data.Region = pm25data.stationId.map(lambda x: x + '#Value#')
    return data


def wide_fw_x_y_split(wdata, history_length, split_date, for_prediction=False):
    col_indices = wdata.columns.tolist()
    pred_start = split_date + pd.Timedelta(1, unit='D')
    pred_end = pred_start + pd.Timedelta(47, unit='h')
    his_end = split_date - pd.Timedelta(1, unit='h')
    his_start = split_date - pd.Timedelta(history_length, unit='D')
    x = wdata.iloc[:, [0] + list(range(col_indices.index(his_start), col_indices.index(his_end) + 1))].copy()
    y = None
    if not for_prediction:
        y = wdata.iloc[:, [0] + list(range(col_indices.index(pred_start), col_indices.index(pred_end) + 1))].copy()
    return x, y


def extract_median(ws, ldata, split_date):
    window = ldata.loc[(ldata.utc_time >= split_date - pd.Timedelta(ws, unit='D')) & (ldata.utc_time < split_date),
                       AQ_COL_NAMES].copy()
    window['hour'] = window['utc_time'].map(lambda x: x.hour)
    medians = window. \
        groupby(['stationId', 'hour']). \
        agg({'PM2.5': np.median, 'PM10': np.median, 'O3': np.median}). \
        reset_index()
    medians = long_to_wide(medians, index='stationId', columns='hour')
    return medians


def extract_median_features(ldata, window_sizes, split_date, n_thread):
    extract_median_thread = partial(extract_median, ldata=ldata, split_date=split_date)
    pool = Pool(n_thread)
    list_of_medians = pool.map(extract_median_thread, window_sizes)
    pool.terminate()
    pool.close()
    median_features = list_of_medians[0]
    for window_idx, median_df in enumerate(list_of_medians[1:]):
        median_features = pd.merge(left=median_features, right=median_df, on='stationId',
                                   suffixes=['', 'mf_window_{}'.format(window_idx)])
    return median_features


def wide_make_fw_x_y(wdata, history_length, split_date, num_shifts=1, shift_step=1, use_medians=False, ldata=None,
                     window_sizes=None, median_shifts=None, use_indicators=False, for_prediction=False, n_thread=1,
                     vprint=print, save_feature=True, use_cache=True, window_name=None):
    x_data, y_data = [], []
    split_date = pd.to_datetime(split_date)
    for shift in range(num_shifts):
        feature_name = 'his_{}_spl_{}_med_{}_medshf_{}_fp_{}'.format(history_length, get_date(split_date), use_medians,
                                                                     median_shifts, for_prediction)
        if window_name is not None:
            feature_name += '_{}'.format(window_name)
        if os.path.exists('../features/{}.pkl'.format(feature_name)) and use_cache:
            vprint(2, "loading pickled data")
            x, y = pickle.load(open('../features/{}.pkl'.format(feature_name), 'rb'))
            x_data.append(x)
            y_data.append(y)
        else:
            vprint(2, '# ---- making the {}th shift ----'.format(shift + 1))
            x, y = wide_fw_x_y_split(wdata=wdata, history_length=history_length, split_date=split_date,
                                     for_prediction=for_prediction)
            vprint(1, 'X: split {} shift {} from {} to {}'.format(split_date, shift, x.columns[1], x.columns[-1]))
            x.columns = ['stationId'] + ['d{}_h{}'.format(d, h) for d in range(history_length) for h in range(24)]
            if use_medians:
                assert (ldata is not None) and (window_sizes is not None), 'need provide window_sizes and ldata'
                for median_shift in range(median_shifts):
                    x_medians = extract_median_features(ldata=ldata, window_sizes=window_sizes,
                                                        split_date=split_date - pd.Timedelta(median_shift, unit='D'),
                                                        n_thread=n_thread)
                    x = pd.merge(left=x, right=x_medians, on='stationId', how='left')

            x['stationId'] = x['stationId'].map(lambda d: d + str(split_date).split(' ')[0])
            x_data.append(x)

            if not for_prediction:
                y['stationId'] = y['stationId'].map(lambda x: x + str(split_date).split(' ')[0])
                vprint(1, 'y: split {} shift {} from {} to {}'.format(split_date, shift, y.columns[1], y.columns[-1]))
                y.columns = ['stationId'] + ['d{}_h{}'.format(d, h)
                                             for d in [i + history_length for i in [1, 2]]
                                             for h in range(24)]
                y_data.append(y)

            if save_feature:
                vprint(2, 'dump feature to pickle file')
                os.system('mkdir -p ../features')
                with open('../features/{}.pkl'.format(feature_name), 'wb') as f:
                    pickle.dump([x, y], f)
        split_date -= pd.Timedelta(shift_step, unit='D')

    x_data = pd.concat(x_data, axis=0)
    y_data = pd.concat(y_data, axis=0) if not for_prediction else None

    if use_indicators:
        x_data['station'] = x_data['stationId'].map(lambda x: x.split('#')[0])
        x_data['measure'] = x_data['stationId'].map(lambda x: x.split('#')[1])
        x_data['city'] = x_data['station'].map(lambda x: 1 if x in LONDON_STATIONS else 0)
        x_measure_dummies = pd.get_dummies(x_data['measure'])
        x_data = pd.concat(objs=[x_data, x_measure_dummies],axis=1)
        x_data.drop(labels=['station', 'measure'], axis=1, inplace=True)
    return x_data, y_data


def standardize_data(data: pd.DataFrame, vprint=print):
    vprint(2, '# ---- standardizing data to zero mean unit variance')
    df = data.copy()  # type: pd.DataFrame
    scalers = {}
    stations = df.stationId.map(lambda x: x.split('#')[0]).unique()
    for station in stations:
        measures = ['PM2.5', 'PM10', 'O3']
        if station in LONDON_STATIONS:
            measures.remove('O3')
        for measure in measures:
            key = '#'.join([station, measure])
            values = df.loc[df['stationId'].map(lambda x: x.split('#')[0] == station), measure]
            scalers[key] = StandardScaler().fit(values)
            df.loc[df['stationId'].map(lambda x: x.split('#')[0] == station), measure] = scalers[key].transform(values)
    return df, scalers


def normalize_datetime(df, keys=('stationId', 'utc_time'), min_time=None, max_time=None):
    assert all(key in df.columns for key in keys), 'some key not in columns'

    if not min_time:
        min_time = df.utc_time.min()
    if not max_time:
        max_time = df.utc_time.max()
    df.drop_duplicates(subset=keys, keep='first', inplace=True)
    stations = df.stationId.unique()
    utc_times = []
    current = min_time
    while current <= max_time:
        utc_times.append(current)
        current += pd.Timedelta(1, 'h')
    left = pd.DataFrame(pd.Series([s, t]) for s in stations for t in utc_times)
    left.columns = ['stationId', 'utc_time']
    if left.shape[0] != df.shape[0]:
        df = pd.merge(left=left, right=df, on=['stationId', 'utc_time'], how='left')
    df.sort_values(by=['stationId', 'utc_time'], inplace=True)
    return df


def get_date_col_index(dataframe, day):
    idx = dataframe.columns.tolist().index(day)
    if idx != len(dataframe.columns):
        return idx
    return None

''' Ham nay lay du lieu tu 28/12/2018 den ngay hom qua va noi lai'''
def download_aq_data_to_concat():
    "Ham nay lay tat ca du lieu muc do o nhiem tu 07/01/2019 den ngay hom nay"
    # Ngay dau tien lay du lieu
    start_time = datetime.date(2019, 1, 7)
    # Ngay cuoi cung lay du lieu la ngay hom nay
    end_time = datetime.date.today()
    # Khoi tao danh sach ngay lay du lieu
    danh_sach_ngay_lay_du_lieu = [start_time + datetime.timedelta(days=x)
                                  for x in range((end_time-start_time).days + 1)]
    # Du lieu tung vung
    final_df = pd.DataFrame(columns=constants.AQ_COL_NAMES);
    for ngay_lay_du_lieu in danh_sach_ngay_lay_du_lieu:
        # Tinh toan ngay lay du lieu hien tai
        date = str(ngay_lay_du_lieu.day) + '/' + str(ngay_lay_du_lieu.month) + '/' + \
            str(ngay_lay_du_lieu.year);
        # Lan luot lay du lieu tung ngay
        url = 'https://www.haze.gov.sg/resources/historical-readings/GetData/{}'.format(date)
        request=requests.get(url)
        # Phan tich ket qua tra ve
        assert request.status_code == 200, 'Network error, status code = ' + str(request.status_code)
        data = request.json()
        # Toan bo du lieu json cua ngay hom do
        airQuality = data['AirQualityList']
        if airQuality == []:
            # Neu khong co du lieu thi tiep tuc
            continue
        for i in airQuality:
            # Chuan hoa thoi gian lay du lieu
            time_text = i['Time']['Text'].split(':')[0]
            if 'pm' in i['Time']['Text']:
                time_text = int(time_text) + 12
            i['Time']['Text'] = '{} {}:00:00'.format(ngay_lay_du_lieu.strftime('%Y-%m-%d'),time_text)
            i['utc_time'] = i['Time']['Text']
            # Luu du lieu cac vung lai
            df = pd.DataFrame([[i['utc_time'], 'North', i['MaxReading']['North']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'South', i['MaxReading']['South']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'East', i['MaxReading']['East']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'West', i['MaxReading']['West']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'Central', i['MaxReading']['Central']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
        print(str(ngay_lay_du_lieu))
    # Sap xep lai thu tu cac cot
    final_df = final_df[constants.AQ_COL_NAMES]
    # Luu vao file .csv
    final_df.to_csv('../test-json-new.csv', index=False)
#     # Doc du lieu tu file json da luu truoc do
#     sing_his_2 = pd.read_csv('../input/Singapore_aq_28_12_2018.csv')
#     # Sap xep lai thu tu cac cot
#     sing_his_2.columns = btl.constants.AQ_COL_NAMES
#     sing_his_2 = sing_his_2[btl.constants.AQ_COL_NAMES]
#     # Ghep 2 file du lieu
#     sing_his = pd.concat([sing_his_2, sing_his_1], axis = 0, sort=True)
#     # Sap xep lai thu tu cac cot
#     sing_his = sing_his[btl.constants.AQ_COL_NAMES]
#     # Luu vao file .csv
#     sing_his.to_csv('../test-json.csv', index=False)
    # Tra ve file du lieu vua ghep
    return final_df
	
'''Tai du lieu ve'''
def download_aq_data():
    "Ham nay lay tat ca du lieu muc do o nhiem tu 01/01/2009 den ngay hom qua"
    # Ngay dau tien lay du lieu
    start_time = datetime.date(2009, 1, 1)
    # Ngay cuoi cung lay du lieu la ngay hom qua
    end_time = datetime.date.today() - timedelta(1)
    # Khoi tao danh sach ngay lay du lieu
    danh_sach_ngay_lay_du_lieu = [start_time + datetime.timedelta(days=x)
                                  for x in range((end_time-start_time).days + 1)]
    # Bien luu du lieu
    final_df = pd.DataFrame(columns=constants.AQ_COL_NAMES);
    # Duyet tung ngay de lay du lieu
    for ngay_lay_du_lieu in danh_sach_ngay_lay_du_lieu:
        # Tinh toan ngay lay du lieu hien tai
        date = str(ngay_lay_du_lieu.day) + '/' + str(ngay_lay_du_lieu.month) + '/' + \
            str(ngay_lay_du_lieu.year);
        # Lan luot lay du lieu tung ngay
        url = 'https://www.haze.gov.sg/resources/historical-readings/GetData/{}'.format(date)
        # Gui get request len server
        request=requests.get(url)
        # Phan tich ket qua tra ve
        assert request.status_code == 200, 'Network error, status code = ' + str(request.status_code)
        # Phan tich json tra ve
        data = request.json()
        # Toan bo du lieu json cua ngay hom do
        airQuality = data['AirQualityList']
        if airQuality == []:
            # Neu khong co du lieu thi tiep tuc
            continue
        for i in airQuality:
            # Chuan hoa thoi gian lay du lieu
            time_text = i['Time']['Text'].split(':')[0]
            if 'pm' in i['Time']['Text']:
                if '12' in i['Time']['Text']:
                    time_text = 12
                else:
                    time_text = int(time_text) + 12
            else:
                if '12' in i['Time']['Text']:
                    time_text = 00
            i['Time']['Text'] = '{} {}:00:00'.format(ngay_lay_du_lieu.strftime('%Y-%m-%d'),time_text)
            i['utc_time'] = i['Time']['Text']
            # Chen du lieu cac vung vao bien luu du lieu
            df = pd.DataFrame([[i['utc_time'], 'North', i['MaxReading']['North']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'South', i['MaxReading']['South']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'East', i['MaxReading']['East']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'West', i['MaxReading']['West']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
            df = pd.DataFrame([[i['utc_time'], 'Central', i['MaxReading']['Central']]], columns=constants.AQ_COL_NAMES)
            final_df = final_df.append(df, ignore_index=True)
        # Cu 1 thang thi luu di lieu lai 1 lan
        tomorrow = ngay_lay_du_lieu - timedelta(1)
        if(tomorrow.month != ngay_lay_du_lieu.month):
            final_df.to_csv('input/sing_his_temp.csv', index=False)
        print(str(ngay_lay_du_lieu))
    # Sap xep lai thu tu cac cot
    final_df = final_df[constants.AQ_COL_NAMES]
    # Luu vao file .csv
    final_df.to_csv('input/sing_his_final.csv', index=False)
    
def get_city_data(*, city: str, impute_with_lgbm: bool=False, partial_data: bool=False, vprint=print,
                  get_new_data: bool=False) -> pd.DataFrame:
    assert city in ['bj', 'ld'], 'invalid city'
    stations = BEIJING_STATIONS if city == 'bj' else LONDON_STATIONS
    end_date = get_date(pd.to_datetime(datetime.datetime.now()))
    vprint(1, '# ---- getting data for {}'.format(city))
    vprint(2, 'loading history data')
    his = pd.read_csv(filepath_or_buffer='../input/{}_api_his.csv'.format(city), parse_dates=['utc_time'])
    if get_new_data:
        vprint(2, 'loading new data')
        new = download_aq_data(city=city, start_date='2018-04-01', start_hour='00', end_date=end_date, end_hour='23',
                               save=False, partial_data=partial_data, data_source='alternative', vprint=vprint)
        data = pd.concat([his, new], axis=0)
    else:
        data = his
    data = data.loc[data['stationId'].isin(stations)][AQ_COL_NAMES]
    data = fix_nat(data)
    if impute_with_lgbm:
        data = lgbm_impute(data=data, city=city, vprint=vprint)
    return data


def evaluate(*, city: str, truth: pd.DataFrame, predictions: pd.DataFrame, measures=None) -> dict:
    scores = dict()
    stations = BEIJING_STATIONS if city == 'bj' else LONDON_STATIONS
    if not measures:
        measures = ['PM2.5', 'PM10', 'O3'] if city == 'bj' else ['PM2.5', 'PM10']
    merged = pd.merge(left=truth, right=predictions, how='left', on='test_id', suffixes=['_ans', '_pred'])
    for station in stations:
        for measure in measures:
            score = official_smape(
                merged.loc[merged['test_id'].map(lambda x: x.split('#')[0] == station)][measure + '_ans'],
                merged.loc[merged['test_id'].map(lambda x: x.split('#')[0] == station)][measure + '_pred']
            )
            scores['{}-{}'.format(station, measure)] = score
    for key in scores:
        if scores[key] == 2:
            scores[key] = np.nan
    return scores


def get_truth(*, city: str, data: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    truth = data.loc[(data['utc_time'] >= start_date) &
                     (data['utc_time'] < start_date + pd.Timedelta(value=2, unit='D'))].copy()  # type: pd.DataFrame
    truth['test_id'] = truth['stationId'] + ['#' + str(i) for i in range(48)] * int(truth.shape[0] / 48)
    if city == 'ld':
        truth['O3'] = 0
    truth.drop(labels=['stationId', 'utc_time'], axis=1, inplace=True)
    truth.dropna(inplace=True)
    return truth


def submit(subfile: str, description: str, filename: str):
    assert os.path.exists(subfile), 'submission file does not exist'
    files = {'files': open(subfile, 'rb')}
    data = {
        "user_id": USERNAME,
        "team_token": SUB_TOKEN,
        "description": description,
        "filename": filename,
    }
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data)
    print(response.text)
