'''
Created on Dec 27, 2018

@author: osboxes
'''
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import requests
import datetime
import btl.MyUtil
# Lay du lieu o nhiem cua singapore
sing_his = btl.MyUtil.download_aq_data()
print('download_aq_data_to_concat done!');
# ngay_lay_du_lieu = datetime.date(2009, 1, 1)
# request=requests.get('https://www.haze.gov.sg/resources/historical-readings/GetData/01/01/2015')
# data = request.json()
# # Toan bo du lieu json cua ngay hom do
# airQuality = data['AirQualityList']
# # Xoa nhung cot khong co du lieu di
# for i in airQuality:
#     del i['PM25Reading']
#     del i['Date']
#     del i['Time']['Value']
# # Thay doi vi tri cac cot
# my_index = ['Time.Text', 'Day', 'Id', 'MaxReading.North', 'MaxReading.South', 'MaxReading.East',\
#          'MaxReading.West', 'MaxReading.Central', 'MaxReading.OverallReading']
# # Ghi cac dong du lieu vao file
# df = pd.DataFrame.from_dict(json_normalize(data["AirQualityList"]), orient='columns',)
# df['Day'] = '01/01/2015'
# df = df[my_index]
# # Luu vao file .csv
# df.to_csv('../test-json.csv', index=False)