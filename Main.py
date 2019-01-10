'''
Created on Dec 27, 2018

@author: osboxes
'''
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import requests
import datetime
import MyUtil
# Lay du lieu o nhiem cua singapore
MyUtil.download_aq_data()
print('download_aq_data done!');