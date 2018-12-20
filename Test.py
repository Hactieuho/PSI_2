'''
Created on Dec 11, 2018

@author: osboxes
'''
import requests
import csv
import re
import datetime
from datetime import date, timedelta

def UpDateDuLieu(fileName, date):
    "Ham nay cap nhat du lieu muc do o nhiem tu ngay trong file den ngay hom qua"
    # Mo file de bat dau doc
    testFileR = open(fileName, 'r', newline='')
    rd = csv.reader(testFileR)
    # Doc den hang cuoi cung xem ngay la bao nhieu
    ngayCuoi = ''
    for hang in rd:
        if re.match(r'\d+/\d+/\d+', hang[0]):
            print(hang[0])
            ngayCuoi = hang[0]
    print('Ngay cuoi: ' + ngayCuoi)
    tachNgayThang = re.split('/', ngayCuoi)
    ngay = int(tachNgayThang[0])
    thang = int(tachNgayThang[1])
    nam = int(tachNgayThang[2])
    print('Ngay: ' + str(ngay))
    print('Thang: ' + str(thang))
    print('Nam: ' + str(nam))
    # Dong file sau khi doc
    testFileR.close();
    
    # Mo file de bat dau chen
    testFileA = open(fileName, 'a', newline='')
    ap = csv.writer(testFileA)
    # Lay tat ca cac ngay trong khoang tu sau ngay cuoi den ngay hom qua
    d1 = datetime.date(nam, thang, ngay) + timedelta(1)
    d2 = date.today() - timedelta(1)
    
    days = [d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)]
    for day in days:
        date = str(day.day) + '/' + str(day.month) + '/' + str(day.year);
        print('Read ' + date)
        # Ghi ngay lay du lieu vao file
        ap.writerow([date])
        # Lay du lieu tren web de phan tich
        req = requests.get('https://www.haze.gov.sg/resources/historical-readings/GetData/' + date)
        # Toan bo du lieu json cua ngay hom do
        airQuality = req.json()['AirQualityList'];
        # Ghi cac dong du lieu vao file
        for i in airQuality:
            dongDuLieu = []
            # Luu lai gio trong ngay
            dongDuLieu += [i['Time']['Text']]
            # Luu lai du lieu trong gio do
            for j in i['MaxReading']:
                dongDuLieu += [i['MaxReading'][j]]
            # Ghi du lieu vao file
            ap.writerow(dongDuLieu)
    # Dong file sau khi ghi
    testFileA.close();

def TaiVeTatCa():
    "Ham nay lay tat ca du lieu muc do o nhiem tu 21/06/2013 den ngay hom qua"
    # Mo file de bat dau ghi
    testFile = open('test.csv', 'w', newline='')
    wr = csv.writer(testFile)
    firstHeader = [''] + ['North'] + ['South'] + ['East'] + ['West'] + ['Central'] + ['OverallReading']
    # Ghi header dau tien vao file vao file
    wr.writerow(firstHeader)
    # Lay tat ca cac ngay trong khoang tu 21/06/2013 den ngay hom qua
    d1 = datetime.date(2013, 6, 21)
    d2 = date.today() - timedelta(1)
    
    days = [d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)]
    for day in days:
        date = str(day.day) + '/' + str(day.month) + '/' + str(day.year);
        print(date)
        # Ghi ngay lay du lieu vao file
        wr.writerow([date])
        # Lay du lieu tren web de phan tich
        req = requests.get('https://www.haze.gov.sg/resources/historical-readings/GetData/' + date)
        # Toan bo du lieu json cua ngay hom do
        airQuality = req.json()['AirQualityList'];
        # Ghi cac dong du lieu vao file
        for i in airQuality:
            dongDuLieu = []
            # Luu lai gio trong ngay
            dongDuLieu += [i['Time']['Text']]
            # Luu lai du lieu trong gio do
            for j in i['MaxReading']:
                dongDuLieu += [i['MaxReading'][j]]
            # Ghi du lieu vao file
            wr.writerow(dongDuLieu)
    # Dong file sau khi ghi
    testFile.close();

UpDateDuLieu('Test.csv', date)