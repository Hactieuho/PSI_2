#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
from constants import REGION, AQ_COL_NAMES
from sklearn.model_selection import GridSearchCV
from MyUtil import download_aq_data, fix_nat, get_date, lgbm_impute,\
    download_aq_data_to_concat
from datetime import datetime
import pickle


def lgb_cv(df, regionId, measure):
    regions = df['Region'].unique()
    data = [df.loc[df['Region'] == region][measure].values for region in regions]
    data = pd.DataFrame(np.array(data)).T
    data.columns = regions
    y = data.loc[~data[regionId].isnull()][regionId]
    X = data.loc[~data[regionId].isnull()][[col for col in regions if col != regionId]]
    lgb_model = lgb.LGBMRegressor(objective='regression', n_jobs=1)
    params_dist = {
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [27, 31, 35],
        'n_estimators': [80, 100],
        'subsample': [0.3, .5, .7],
        'colsample_bytree': [0.3, .5, .7],
        'reg_alpha': [0, .2],
        'reg_lambda': [0, .2]
    }
    results = GridSearchCV(
        estimator=lgb_model,
        param_grid=params_dist,
        scoring='neg_mean_absolute_error',
        n_jobs=8,
        cv=5,
        refit=True,
        verbose=1,
        return_train_score=True
    ).fit(X, y)
    print(results.best_params_)
    return results.best_estimator_


def main(mode: str='train'):
    if mode == 'train':
        # Tao thu muc models
        os.system('mkdir -p models')
        # Mo file data
        assert os.path.exists('input/sing_aq_history.csv'), 'run btl.MyUtil.download_aq_data() to get data first'
        sing_his = pd.read_csv(filepath_or_buffer='input/sing_aq_history.csv', parse_dates=['utc_time'])
        # Dictionary
        bj_lgb_models = {
            '{}-{}'.format(region, 'Value'): lgb_cv(sing_his, region, 'Value')
            for region in REGION
        }
        with open("models/bj_lgbm.pkl", 'wb') as f:
            pickle.dump(bj_lgb_models, f)
        print('# ---- DONE ---- #')

    if mode == 'impute':
        assert os.path.exists('models/bj_lgbm.pkl'), 'model not trained yet'
        bj_his = pd.read_csv(filepath_or_buffer='input/sing_aq_history.csv', parse_dates=['utc_time'])
        bj_new = download_aq_data_to_concat()
        bj_new = bj_new.loc[bj_new.utc_time < pd.to_datetime('today') - pd.Timedelta(1, 'D')]
        bj_data = pd.concat([bj_his, bj_new], axis=0)
        bj_data = bj_data.loc[bj_data.stationId.isin(REGION)]
        bj_data = bj_data[AQ_COL_NAMES]
        bj_data = fix_nat(bj_data)

        bj_data = lgbm_impute(data=bj_data, city='bj')
        data = pd.concat([bj_data], axis=0)
        data = fix_nat(data)
        data.to_csv('input/lgb_imputed_new_source_2014-03-31-_{}.csv'.format('today'), index=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        mode = 'impute'
    else:
        mode = str(sys.argv[1])
    assert mode in ['train', 'impute'], 'invalid mode'
    main(mode)
