import time
import random
import threading
from PIL import Image
from flask import Flask, send_file, render_template, jsonify

import json, datetime, os,re
import numpy as np
import warnings
import time, requests
import pandas as pd
import winsound
from playsound import playsound
import matplotlib.pyplot as plt
from collections import deque
from utils.EM_fields_202303 import EM_fields

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
import utils.tdxExhq_config as conf
from pytdx.params import TDXParams

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# pd.describe_option()
pd.options.mode.chained_assignment = None
pd.set_option('precision', 6)
pd.set_option('display.precision', 4)
pd.set_option('display.float_format',  '{:.2f}'.format)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('expand_frame_repr', False)
warnings.filterwarnings('ignore')
np.random.seed(42)


app = Flask(__name__)

def TestConnection(Api, type, ip, port):
    if type == 'HQ':
        try:
            is_connect = Api.connect(ip, port)
        except Exception as e:
            print('Failed to connect to HQ!')
            exit(0)

        if is_connect is False:  # 失败了返回False，成功了返回地址
            print('HQ is_connect is False!')
            return False
        else:
            print('HQ is connected!')
            return True

    elif type=='ExHQ':
        try:
            is_connect = Api.connect(ip, port)
        except Exception as e:
            print('Failed to connect to Ext HQ!')
            exit(0)

        if is_connect is False:  # 失败了返回False，成功了返回地址
            print('ExHQ is_connect is False!')
            return False
        else:
            print('ExHQ is connected')
            return True

def playalarm():
    # print('Alarm!!!!')
    playsound('D:\\mp3\\effect\\miao.mp3')

def cal_right_price(input_stock_data, type='前复权'):
    """
    :param input_stock_data: 标准股票数据，需要'收盘价', '涨跌幅'
    :param type: 确定是前复权还是后复权，分别为'后复权'，'前复权'
    :return: 新增一列'后复权价'/'前复权价'的stock_data
    """
    # 计算收盘复权价
    stock_data = input_stock_data.copy()
    num = {'后复权': 0, '前复权': -1}

    price1 = stock_data['close'].iloc[num[type]]
    stock_data['复权价_temp'] = (stock_data['change'] + 1.0).cumprod()
    price2 = stock_data['复权价_temp'].iloc[num[type]]
    stock_data['复权价'] = stock_data['复权价_temp'] * (price1 / price2)
    stock_data.pop('复权价_temp')

    # 计算开盘复权价
    stock_data['复权价_开盘'] = stock_data['复权价'] / (stock_data['close'] / stock_data['open'])
    stock_data['复权价_最高'] = stock_data['复权价'] / (stock_data['close'] / stock_data['high'])
    stock_data['复权价_最低'] = stock_data['复权价'] / (stock_data['close'] / stock_data['low'])

    return stock_data[['复权价_开盘', '复权价', '复权价_最高', '复权价_最低']]

def readDapanData(startdate,enddate):

    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}
    df_sh = pd.read_csv('D:\\stockstudy\\win2021\\baodata\\bao-sh.000001.csv',dtype=dtype_dic, encoding='gbk', )
    df_sh = df_sh[(df_sh['date']>=startdate) & (df_sh['date']<=enddate)]
    df_sz = pd.read_csv('D:\\stockstudy\\win2021\\baodata\\bao-sz.399001.csv',dtype=dtype_dic, encoding='gbk', )
    df_sz = df_sz[(df_sh['date']>=startdate) & (df_sz['date']<=enddate)]

    return df_sh,df_sz

def load_file(path, file,start_time,end_time):
    try:
        path += file
        df = pd.read_csv(path, encoding='gbk', skiprows=0) # parse_dates=['交易日期']
        # df = df[['股票代码','股票名称','交易日期','开盘价','最高价','最低价','收盘价','前收盘价']]
        df.columns = ['date','code','开盘价','最高价','最低价','收盘价','volume','amount','turn','pctChg']
        df['change'] = df['pctChg']/100
        # df['date'] = df['交易日期']
        df = df[(df['date']>=start_time) & (df['date']<=end_time)]
        df[['open', 'close', 'high', 'low']] = cal_right_price(df, type='后复权')
        return df
    except:
        print('load data file failed')
        return pd.DataFrame()

def getNorthzjlx():
    url = 'https://push2.eastmoney.com/api/qt/kamtbs.rtmin/get?fields1=f1,f2,f3,f4&fields2=f51,f54,f52,f58,f53,f62,f56,f57,f60,f61&'+ \
          'ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery112309012195417245222_1655353679439&_=1655353679440'

    # res = requests.get(url)
    res = requests.get(url)

    try:
        data1 = json.loads(res.text[42:-2])['data']['s2n']
    except:
        return pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1],columns=['time', 'hgtnet', 'hgtin', 'sgtnet', 'hgtout', 'north', 'sgtin','sgtout', 'northin', 'northout'])
    min.drop(labels=['time', 'hgtnet', 'hgtin', 'sgtnet', 'hgtout', 'sgtin','sgtout', 'northin', 'northout'],axis=1,inplace=True)
    min = min[min['north']!='-']
    # min['time'] = min['time'].astype('datetime64[ns]')
    min['north'] = min['north'].astype('float')/10000
    min['northdelta'] = min['north'] - min['north'].shift(1)
    min['northdeltam5'] = min['northdelta'].rolling(5).mean()

    return min

def getStockBaseInfo(code):

    if len(code)==6 and code[0] in ['6']:
        emcode = '1.' + code
    else:
        emcode = '0.' + code
    url = 'http://push2.eastmoney.com/api/qt/stock/get?invt=2&fltt=1&cb=jQuery35101209717597227119_1679475825970&' \
          'ut=fa5fd1943c7b386f172d6893dbfba10b&wbp2u=|0|0|0|web&_=1679475825971'
    fields = 'f58,f57,f3,f2'
    params = {'secid': emcode}#, 'fieldssssss':fields}
    res = requests.get(url,params=params)

    try:
        restext = re.search('{.*}',res.text).group()
        data1 = json.loads(restext)['data']
        data = pd.DataFrame(data1,index=[0])
        cols = [EM_fields[f] for f in data.columns if f in EM_fields.keys()]
        data.rename(columns=EM_fields)
    except:
        return pd.DataFrame()


    return min

def getHS300zjlx():
    url = 'https://push2.eastmoney.com/api/qt/stock/fflow/kline/get?cb=jQuery112304151702712546592_1652363383462&lmt=0&klt=1'+ \
          '&fields1=f1%2Cf2%2Cf3%2Cf7&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61%2Cf62%2Cf63%2Cf64%2Cf65&ut=b2884a393a59ad64002292a3e90d46a5'+ \
          '&secid=90.BK0500&_=1652363383463'
    res = requests.get(url)

    try:
        data1 = json.loads(res.text[42:-2])['data']['klines']
    except:
        return pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1],columns=['time', 'boss', 'small', 'med', 'big', 'huge'])
    min.drop(labels=['time','small','med','big','huge'],axis=1,inplace=True)
    # min['time'] = min['time'].astype('datetime64[ns]')
    min['boss'] = min['boss'].astype('float')/100000000
    min['net'] = min['boss'] - min['boss'].shift(1)
    min['bossma5'] = min['net'].rolling(5).mean()

    return min

def MINgetDPindex():
    global AmtFactor

    url = 'http://push2his.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery112409887318613050171_1615514818501&secid=1.000001&secid2=0.399001'+ \
          '&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6%2Cf7%2Cf8%2Cf9%2Cf10%2Cf11&fields2=f51%2Cf53%2Cf56%2Cf58&iscr=0&ndays=1&_=1615514818608'
    res = requests.get(url)
    try:
        data1 = json.loads(res.text[42:-2])['data']['trends']
        preclose = json.loads(res.text[42:-2])['data']['preClose']
    except:
        return pd.DataFrame()
    data = pd.DataFrame([i.split(',') for i in data1], columns=['time', 'close', 'vol', 'mean'])
    data.drop(labels=['mean'],axis=1,inplace=True)
    # min.rename(columns={'mea'})
    # data['time'] = data['time'].astype('datetime64[ns]')
    data['close'] = data['close'].astype('float')
    data['vol'] = data['vol'].astype('float')
    data['amt'] = data['close']*data['vol']
    data['amtcum'] = data['amt'].cumsum()
    data['volcum'] = data['vol'].cumsum()
    data['avg'] = data['amtcum']/data['volcum']
    data['amttrend'] = 0
    factor = [AmtFactor[0]] + AmtFactor
    # factor = [30.7395,20.5736,15.8613,13.0092,11.1578,9.8177,8.8125,8.0707,7.4616,6.9545,6.4752,6.0869,
    #           5.7620,5.4838,5.2396,5.0007,4.7964,4.6182,4.4579,4.3133,4.1607,4.0341,3.9201,3.8174,
    #           3.7208,3.6250,3.5401,3.4608,3.3838,3.3129,3.2326,3.1652,3.1035,3.0456,2.9930,2.9433,
    #           2.8948,2.8507,2.8081,2.7674,2.7227,2.6848,2.6485,2.6142,2.5811,2.5484,2.5176,2.4888,
    #           2.4612,2.4350,2.4049,2.3779,2.3535,2.3304,2.3081,2.2867,2.2657,2.2459,2.2266,2.2078,
    #           2.1842,2.1634,2.1444,2.1266,2.1095,2.0926,2.0763,2.0606,2.0453,2.0305,2.0136,1.9986,
    #           1.9844,1.9709,1.9579,1.9447,1.9320,1.9200,1.9083,1.8966,1.8839,1.8728,1.8619,1.8512,
    #           1.8409,1.8307,1.8206,1.8108,1.8013,1.7921,1.7813,1.7713,1.7617,1.7523,1.7435,1.7346,
    #           1.7260,1.7179,1.7099,1.7021,1.6932,1.6849,1.6772,1.6699,1.6626,1.6550,1.6477,1.6403,
    #           1.6332,1.6264,1.6189,1.6119,1.6053,1.5991,1.5929,1.5870,1.5811,1.5756,1.5701,1.5641,
    #           1.5484,1.5415,1.5353,1.5292,1.5231,1.5168,1.5105,1.5047,1.4989,1.4931,1.4864,1.4803,
    #           1.4743,1.4684,1.4629,1.4574,1.4518,1.4464,1.4408,1.4351,1.4286,1.4228,1.4172,1.4117,
    #           1.4062,1.4007,1.3953,1.3897,1.3841,1.3785,1.3718,1.3664,1.3612,1.3561,1.3511,1.3464,
    #           1.3416,1.3369,1.3323,1.3277,1.3227,1.3179,1.3133,1.3088,1.3044,1.2999,1.2955,1.2911,
    #           1.2868,1.2826,1.2779,1.2735,1.2694,1.2653,1.2613,1.2571,1.2532,1.2494,1.2456,1.2418,
    #           1.2371,1.2328,1.2289,1.2251,1.2214,1.2176,1.2137,1.2101,1.2065,1.2029,1.1990,1.1954,
    #           1.1918,1.1883,1.1848,1.1812,1.1777,1.1743,1.1708,1.1674,1.1636,1.1602,1.1567,1.1533,
    #           1.1498,1.1463,1.1429,1.1396,1.1362,1.1329,1.1287,1.1248,1.1211,1.1173,1.1136,1.1099,
    #           1.1062,1.1024,1.0987,1.0949,1.0907,1.0866,1.0826,1.0786,1.0745,1.0701,1.0659,1.0615,
    #           1.0572,1.0527,1.0475,1.0425,1.0370,1.0313,1.0257,1.0192,1.0124,1.0118,1.0118,1.0,1.0]

    for i in range(len(data)):
        data['amttrend'][i] = data['amtcum'][i] * factor[i]
    data.drop(columns=['amtcum','volcum','vol','amt'], inplace=True)

    data['close'] = data['close'].astype('float')
    data['avg'] = data['avg'].astype('float')

    # data = get_higher_low_up(data, 3)

    return data, preclose

def getETFindex(etfcode):
    if etfcode[0] =='1':
        url = 'http://push2.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery112409911884668744422_1615731971539&secid=0.' + \
              etfcode + '&ut=fa5fd1943c7b386f172d6893dbfba10b&'+ \
              'fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6%2Cf7%2Cf8%2Cf9%2Cf10%2Cf11%2Cf12%2Cf13&'+ \
              'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&iscr=0&ndays=1&_=1615731971555'
    else:
        url = 'http://push2.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery112409911884668744422_1615731971539&secid=1.' + \
              etfcode + '&ut=fa5fd1943c7b386f172d6893dbfba10b&'+ \
              'fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6%2Cf7%2Cf8%2Cf9%2Cf10%2Cf11%2Cf12%2Cf13&'+ \
              'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&iscr=0&ndays=1&_=1615731971555'
    res = requests.get(url)
    try:
        data1 = json.loads(res.text[42:-2])['data']['trends']
        preclose = json.loads(res.text[42:-2])['data']['preClose']
    except:
        return pd.DataFrame(),0

    data = pd.DataFrame([i.split(',') for i in data1], columns=['time', 'ref1', 'close', 'ref2','ref3', 'ref4','vol', 'avg'])
    data.drop(labels=['ref1', 'ref2','ref3', 'ref4'],axis=1,inplace=True)
    data['time'] = data['time'].astype('datetime64[ns]')
    data['close'] = data['close'].astype('float')
    data['vol'] = data['vol'].astype('float')
    data['avg'] = data['avg'].astype('float')

    day = datetime.datetime.now().strftime('%Y-%m-%d')
    df_poscnt = pd.DataFrame(api.get_index_bars(8,1, '880005', 0, 240))
    if len(df_poscnt)==0:
        return pd.DataFrame(),0
    df_poscnt['day'] = df_poscnt['datetime'].apply(lambda x: x[:10])
    df_poscnt = df_poscnt[df_poscnt['day']==day]
    df_poscnt.reset_index(drop=True, inplace=True)
    df_poscnt.rename(columns={'close':'upcnt'}, inplace=True)
    data = pd.merge(data, df_poscnt[['upcnt']], left_index=True, right_index=True,how='left')

    return data,preclose

def MINgetZjlxDP():
    url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=1.000001&secid2=0.399001&' + \
          'fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&' + \
          'ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery18308174687833149541_1607783437004&_=1607783437202'
    res = requests.get(url)

    try:
        data1 = json.loads(res.text[41:-2])['data']['klines']
    except:
        return pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1],columns=['time', 'boss', 'small', 'med', 'big', 'huge'])
    min.drop(labels=['small','med','big','huge'],axis=1,inplace=True)
    min['time'] = min['time'].astype('datetime64[ns]')
    min['boss'] = min['boss'].astype('float')
    min['net'] = min['boss'] - min['boss'].shift(1)
    min['bossma5'] = min['net'].rolling(5).mean()

    return min

def getTrendLine(df, high, low,cmin, cmax):

    data1 = df[df.index>=cmax]
    if 'highs' in data1.columns:
        data1 = data1[data1['highs']==True]
    while len(data1)>3:
        reg0 = linregress(x=data1['date_id'], y=data1[high],  )
        if data1.loc[data1[high]>=reg0[0]*data1['date_id']+reg0[1]].shape[0] == data1.shape[0] or \
                data1.loc[data1[high]>=reg0[0]*data1['date_id']+reg0[1]].shape[0]<3:
            break
        else:
            data1 = data1.loc[data1[high]>=reg0[0]*data1['date_id']+reg0[1]]
    reg0 = linregress(x=data1['date_id'], y=data1[high],  )

    data1 = df[df.index>=cmin]
    if 'lows' in data1.columns:
        data1 = data1[data1['lows']==True]
    while len(data1)>3:
        reg1 = linregress(x=data1['date_id'],   y=data1[low], )
        if data1.loc[data1[low]<=reg1[0]*data1['date_id']+reg1[1]].shape[0] == data1.shape[0] or \
                data1.loc[data1[low]<=reg1[0]*data1['date_id']+reg1[1]].shape[0]<3:
            break
        else:
            data1 = data1.loc[data1[low]<=reg1[0]*data1['date_id']+reg1[1]]
    reg1 = linregress(x=data1['date_id'], y=data1[low], )

    return reg0, reg1

def min2day(timestr):
    global AmtFactor
    factor = AmtFactor
    # factor = [30.7395,20.5736,15.8613,13.0092,11.1578,9.8177,8.8125,8.0707,7.4616,6.9545,6.4752,6.0869,
    #           5.7620,5.4838,5.2396,5.0007,4.7964,4.6182,4.4579,4.3133,4.1607,4.0341,3.9201,3.8174,
    #           3.7208,3.6250,3.5401,3.4608,3.3838,3.3129,3.2326,3.1652,3.1035,3.0456,2.9930,2.9433,
    #           2.8948,2.8507,2.8081,2.7674,2.7227,2.6848,2.6485,2.6142,2.5811,2.5484,2.5176,2.4888,
    #           2.4612,2.4350,2.4049,2.3779,2.3535,2.3304,2.3081,2.2867,2.2657,2.2459,2.2266,2.2078,
    #           2.1842,2.1634,2.1444,2.1266,2.1095,2.0926,2.0763,2.0606,2.0453,2.0305,2.0136,1.9986,
    #           1.9844,1.9709,1.9579,1.9447,1.9320,1.9200,1.9083,1.8966,1.8839,1.8728,1.8619,1.8512,
    #           1.8409,1.8307,1.8206,1.8108,1.8013,1.7921,1.7813,1.7713,1.7617,1.7523,1.7435,1.7346,
    #           1.7260,1.7179,1.7099,1.7021,1.6932,1.6849,1.6772,1.6699,1.6626,1.6550,1.6477,1.6403,
    #           1.6332,1.6264,1.6189,1.6119,1.6053,1.5991,1.5929,1.5870,1.5811,1.5756,1.5701,1.5641,
    #           1.5484,1.5415,1.5353,1.5292,1.5231,1.5168,1.5105,1.5047,1.4989,1.4931,1.4864,1.4803,
    #           1.4743,1.4684,1.4629,1.4574,1.4518,1.4464,1.4408,1.4351,1.4286,1.4228,1.4172,1.4117,
    #           1.4062,1.4007,1.3953,1.3897,1.3841,1.3785,1.3718,1.3664,1.3612,1.3561,1.3511,1.3464,
    #           1.3416,1.3369,1.3323,1.3277,1.3227,1.3179,1.3133,1.3088,1.3044,1.2999,1.2955,1.2911,
    #           1.2868,1.2826,1.2779,1.2735,1.2694,1.2653,1.2613,1.2571,1.2532,1.2494,1.2456,1.2418,
    #           1.2371,1.2328,1.2289,1.2251,1.2214,1.2176,1.2137,1.2101,1.2065,1.2029,1.1990,1.1954,
    #           1.1918,1.1883,1.1848,1.1812,1.1777,1.1743,1.1708,1.1674,1.1636,1.1602,1.1567,1.1533,
    #           1.1498,1.1463,1.1429,1.1396,1.1362,1.1329,1.1287,1.1248,1.1211,1.1173,1.1136,1.1099,
    #           1.1062,1.1024,1.0987,1.0949,1.0907,1.0866,1.0826,1.0786,1.0745,1.0701,1.0659,1.0615,
    #           1.0572,1.0527,1.0475,1.0425,1.0370,1.0313,1.0257,1.0192,1.0124,1.0118,1.0118,1.0]
    hr, min,sec = map(int, timestr.split(':'))
    tradingmin = hr*60+min - 9*60-30
    if tradingmin>0 and tradingmin<120:
        return factor[tradingmin-1]
    elif tradingmin>=120 and tradingmin<=210:
        return factor[119]
    elif tradingmin>210 and tradingmin<330:
        return factor[tradingmin-91]
    else:
        return 1

def calAmtFactor(n):
    df_day = pd.DataFrame(api.get_index_bars(9, 1, '999999', 0, n+1))
    df_day['date'] = df_day['datetime'].apply(lambda x: x[:10])
    daylist = df_day['date'].values[:-1]

    times = (n+1)*240//800+1 if (n+1)*240%800>0 else (n+1)*240//800
    df_min = pd.DataFrame()
    for i in range(times-1,-1,-1):
        temp = pd.DataFrame(api.get_index_bars(8, 1, '999999', i*800, 800))
        if len(temp)>0:
            df_min = pd.concat([df_min, temp])

    df_min['date'] = df_min['datetime'].apply(lambda x: x[:10])
    df_min['time'] = df_min['datetime'].apply(lambda x: x[11:])
    df_min = df_min[(df_min['date']>=daylist[0]) & (df_min['date']<=daylist[-1])]
    df_min.reset_index(drop=True,inplace=True)
    amt_all = df_min['amount'].sum()
    df_result = pd.DataFrame(data=df_min['time'].values[0:240], columns=['minute'])
    df_result['minpct'] = 0
    for min in df_min['time'].values[0:240]:
        amt_min = df_min[df_min['time']==min]['amount'].sum()
        df_result.loc[df_result['minute']==min, 'minpct'] = amt_min / amt_all
    df_result['pct'] = df_result['minpct'].cumsum()
    df_result['factor'] = 1/df_result['pct']

    return list(df_result['factor'].values)

def getTdxRealtime(api, market, code):
    # data = api.get_security_quotes([(0, '000001'), (1, '600300')])
    data = api.get_security_quotes([(market, code)])
    if len(data)>0:
        close = data[0]['price']
        preclose = data[0]['last_close']
        high = data[0]['high']
        low = data[0]['low']
        open = data[0]['open']
        timestr = datetime.datetime.now().strftime('%H:%M:%S')
        amount = data[0]['amount']*min2day(timestr)
        vol = data[0]['vol']*min2day(timestr)
        # date, close, amount, vol,high,low,open
        return [datetime.datetime.now().strftime('%Y-%m-%d'), close, amount, vol,high,low,open, preclose]
    else:
        print(str(market)+' '+ code+ " no data.")
        return ['',0,0,0,0,0,0,0]

def pullRealtimeData(code):
    if code[0] == '6':
        code = "1." + code
    elif code[0]=='0' or code[0]=='3':
        code = "0." + code
    elif code[:2].lower()=='sh':
        code = "1." + code[2:].replace('.','')
    elif code[:5].lower()=='sz399':
        code = "0." + code[2:].replace('.','')
    elif code[:6].lower()=='sz.399':
        code = "0." + code[2:].replace('.','')
    else:
        print('code not accepted '+  code)

    url = 'http://push2.eastmoney.com/api/qt/stock/get?ut=fa5fd1943c7b386f172d6893dbfba10b&invt=2&fltt=2&fields=f43,f44,f45,f46,f47,f48,f60,f170&secid=' + \
          code + '&cb=jQuery112404973692212841755_1642991317570&_=1642991317571'
    # host = "hq.sinajs.cn"
    # ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0"
    # my_referer = 'http://finance.sina.com.cn/realstock/company/' + code.replace('.', '') + '/nc.shtml'

    try:
        # res = requests.get(url,headers={'referer': my_referer, 'Host':host, "User-Agent":ua})
        res = requests.get(url)
        if res.status_code == 200 and len(res.text)>40:
            datajson =json.loads(res.text[42:-2])['data']
            open = datajson['f46']
            high = datajson['f44']
            low = datajson['f45']
            close = datajson['f43']
            preclose = datajson['f60']

            # volume = datajson['f47'].replace('-','0')*min2day(datetime.datetime.now().strftime('%H:%M:%S'))
            # amount = datajson['f48'].replace('-','0')*min2day(datetime.datetime.now().strftime('%H:%M:%S'))
            volume, amount = 0,0
            turn = 0
            pctChg = datajson['f170']
            date = datetime.datetime.now().strftime('%Y-%m-%d')

            return [{'date': date, 'code': code, 'open': open, 'high': high, 'low': low, 'close': close,# 'preclose': preclose,
                     'volume': volume, 'amount': amount, 'turn': turn, 'preclose':preclose,
                     'pctChg': pctChg}]
        else:
            print('sEM_error ' +res.text)
            return []

    except requests.exceptions.RequestException:
        print('sina_error '+code)
        return []

def getStk1minV2(code, name, api, Exapi):

    try:
        if '指数' in name:
            if code[0] == '0':
                # data = pd.DataFrame(api.get_history_minute_time_data(TDXParams.MARKET_SH, code, day))
                data = pd.DataFrame(api.get_index_bars(8, TDXParams.MARKET_SH, code, 0,240))
                if len(data)<120:
                    return pd.DataFrame()
            else:
                data = pd.DataFrame(api.get_index_bars(8, TDXParams.MARKET_SH, code, 0,240))
                if len(data)<120:
                    return pd.DataFrame()
            data['date'] = data['datetime'].apply(lambda x: x[11:])
        elif len(code)==6 and code[0] in ['5','6']:
            mkt = 1
            data = pd.DataFrame(api.get_security_bars(8, mkt, code, 0, 240))
            data['date'] = data['datetime'].apply(lambda x: x[11:])
        elif len(code)==6 and code[0] in ['0','3']:
            mkt = 0
            data = pd.DataFrame(api.get_security_bars(8, mkt, code, 0, 240))
            data['date'] = data['datetime'].apply(lambda x: x[11:])
        elif len(code)==6 and code[:2] in ['11']:
            mkt = 1
            data = pd.DataFrame(api.get_security_bars(8, mkt, code, 0, 240))
            data['date'] = data['datetime'].apply(lambda x: x[11:])
        elif len(code)==6 and code[:2] in ['15','12']:
            mkt = 0
            data = pd.DataFrame(api.get_security_bars(8, mkt, code, 0, 240))
            data['date'] = data['datetime'].apply(lambda x: x[11:])
        elif len(code)==5:
            data = pd.DataFrame(Exapi.get_instrument_bars(8, 71, code, 0,200))
            if len(data)<120:
                return pd.DataFrame()
            data['date'] = data['datetime'].apply(lambda x: x[11:])
        else:
            return pd.DataFrame()

        data.reset_index(drop=True, inplace=True)
        return data

    except Exception as e:
        print('exception msg: '+ str(e))
        print(' -----  exception, restart main ----- ')
        time.sleep(5)
        main()

def processSingleStock1min(code,name,api,Exapi):
    global dayr1

    try:

        # df_1min = getStk1min(code, name, api,Exapi)
        df_1min = getStk1minV2(code, name, api,Exapi)
        if len(df_1min) == 0:
            print(code+' '+name +' TdxRealtime failed.')
            return

        df_1min['ma10'] = df_1min['close'].rolling(10).mean()
        df_1min['ma20'] = df_1min['close'].rolling(20).mean()
        df_1min['ma60'] = df_1min['close'].rolling(60).mean()

        n = 3 # number of points to be checked before and after
        df_1min['min'] = df_1min.iloc[argrelextrema(df_1min.close.values, np.less, order=n)[0]]['close']
        df_1min['max'] = df_1min.iloc[argrelextrema(df_1min.close.values, np.greater, order=n)[0]]['close']
        df_1min["isHigh"]= (df_1min["close"]==df_1min['max'])
        df_1min["isLow"]= (df_1min["close"]==df_1min['min'])

        flag = 0
        cnow = df_1min['close'].values[-1]
        cr2 = df_1min['close'].values[-2]
        idxnow = df_1min.index.values[-1]
        high0 = df_1min['close'].max()
        high0idx = df_1min[df_1min['close']==high0].index.values[0]
        low0 = df_1min['close'].min()
        low0idx = df_1min[df_1min['close']==low0].index.values[0]
        # df_1min['date_id'] = [j for j in range(len(df_1min))]

        if high0idx>low0idx:   # 上升中
            if idxnow-high0idx>15: # 绝对高点在15分钟前
                low2 = df_1min[high0idx:]['close'].min()    # 高点右侧低点
                low1 = df_1min[:high0idx][df_1min["close"]==df_1min['min']]['close'].max() # 低点左侧高点

                if cnow<low1 and cr2>low1 and df_1min['close'].values[-1]<df_1min['ma10'].values[-1]:  # cnow<lowThreashold and cr2>lowThreasholdr2 and
                    print('DOWN ' +code + ' '+ name + ' ' + ' 跌破顶部左侧低点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                                   '  止损价:' + str(round(low1,3)) + ' 止损%:' + str(round(low1/df_1min['close'].values[-1]*100-100,2)) + '%')
                    flag = -1
                    # playalarm()
                elif cnow<low2 and cr2>low2 and df_1min['close'].values[-1]<df_1min['ma10'].values[-1]:
                    print('DOWN ' +code + ' '+ name + ' ' + ' 跌破顶部右侧低点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                                   '  止损价:' + str(round(low2,3)) + ' 止损%:' + str(round(low2/df_1min['close'].values[-1]*100-100,2)) + '%')
                    flag = -1
                    # playalarm()
                else:
                    return
            elif idxnow==high0idx and idxnow-df_1min[:-1]['close'].argmax()>=10 and idxnow-low0idx>25 and cnow>df_1min[:-1]['close'].max(): # 刚创新高
                high1idx = df_1min[:-1]['close'].argmax()
                low1 = df_1min[high1idx:]['close'].min()
                print('UP '+code + ' '+ name  + ' ' + ' 上升中刚创新高 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                               '  止损价:' + str(round(low1,3)) + ' 止损%:' + str(round(df_1min['close'].values[-1]/low1*100-100,2)) + '%')
                flag = 2
                playalarm()
            else:
                return
        else:   #  high0idx<low0idx 下跌中
            if idxnow-low0idx>15: # 绝对低点在15分钟前
                high2 = df_1min[low0idx:]['close'].max()    # 低点右侧高点
                high1 = df_1min[:low0idx][df_1min["close"]==df_1min['max']]['close'].min() # 低点左侧高点

                if cnow>high1 and cr2<high1 and df_1min['close'].values[-1]>df_1min['ma10'].values[-1]:  # cnow>highThreashold and cr2<highThreasholdr2
                    print('UP ' +code + ' '+ name  + ' ' + ' 突破底部左侧高点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                                   '  止损价:' + str(round(high1,3)) + ' 止损%:' + str(round(df_1min['close'].values[-1]/high1*100-100,2)) + '%')
                    flag = 1
                    playalarm()
                elif cnow>high2 and cr2<high2 and df_1min['close'].values[-1]>df_1min['ma10'].values[-1]:
                    print('UP ' +code + ' '+ name  + ' ' + ' 突破底部右侧高点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                                   '  止损价:' + str(round(high2,3)) + ' 止损%:' + str(round(df_1min['close'].values[-1]/high2*100-100,2)) + '%')
                    flag = 1
                    playalarm()
                else:
                    return
            elif idxnow==low0idx and idxnow-df_1min[:-1]['close'].argmin()>=10 and idxnow-high0idx>25 and cnow<df_1min[:-1]['close'].min(): # 刚创新低
                low1idx = df_1min[:-1]['close'].argmin()
                high1 = df_1min[low1idx:]['close'].max()
                print('DOWN ' +code + ' '+ name  + ' ' + ' 下跌中刚创新低 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                               '  止损价:' + str(round(high1,3)) + ' 止损%:' + str(round(high1/df_1min['close'].values[-1]*100-100,2)) + '%')
                flag = -2
                # playalarm()
            else:
                return

        return

    except:
        print('processing '+ code+' '+ name+ ' error')
        return

def plotAllzjlx():
    df_north = getNorthzjlx()  # northdelta
    hs300_zjlx = getHS300zjlx()  # zjlxdelta
    hs300_index,hs_preclose = getETFindex('510300')  #   close
    if len(hs300_index) == 0:
        print('failed to get 300ETF df, skipped')
        return
    dp_zjlx = MINgetZjlxDP()   # zjlxdelta
    dp_index, dp_preclose = MINgetDPindex()  # close

    dp_boss = dp_zjlx.boss.values[-1]/100000000
    dp_north = df_north.north.values[-1]
    hs300_boss = hs300_zjlx.boss.values[-1]
    upcnt = hs300_index.upcnt.values[-2]
    df_dp = pd.merge(df_north, dp_zjlx, left_index=True, right_index=True)
    df_dp = pd.merge(df_dp, dp_index, left_index=True, right_index=True)
    df_hs300 = pd.merge(hs300_zjlx, hs300_index,left_index=True, right_index=True)
    dp_amount = str(round(list(df_dp.amttrend)[-1]/100000000,2))

    #################

    df_hs300.reset_index(drop=True,inplace=True)
    df_hs300.reset_index(inplace=True)
    df_hs300['chighlow30'] = df_hs300['close'].rolling(30).max()/df_hs300['close'].rolling(30).min()*100-100
    df_hs300['chighlow60'] = df_hs300['close'].rolling(60).max()/df_hs300['close'].rolling(60).min()*100-100
    df_hs300['cp30'] = (df_hs300['close']-df_hs300['close'].rolling(30).min())/(df_hs300['close'].rolling(30).max()-df_hs300['close'].rolling(30).min())
    df_hs300['cp60'] = (df_hs300['close']-df_hs300['close'].rolling(60).min())/(df_hs300['close'].rolling(60).max()-df_hs300['close'].rolling(60).min())
    df_hs300['cm10'] = df_hs300['close'].rolling(10).mean()
    df_hs300['cabovem10'] = df_hs300['close']> df_hs300['cm10']

    df_hs300.loc[(df_hs300['cp30']<0.3) & (df_hs300['cp60']<0.3) &  (df_hs300['cabovem10']==True) & \
                 (df_hs300['close']>df_hs300['close'].shift(1)), 'up'] = True
    df_hs300.loc[(df_hs300['cp30']>0.7) & (df_hs300['cp60']>0.7) &  (df_hs300['cabovem10']==False), 'dw'] = True

    if df_hs300['up'].values[-1]==True:
        msg = 'UP 510300 hs300' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))
        print('UP 510300 hs300' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = 'http://wx.xtuis.cn/XfhmghWTRzitW6RHalZc8AzN5.send?text=' + msg
        requests.get(msgURL)
        playalarm()
    if df_hs300['dw'].values[-1]==True:
        msg = 'UP 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))
        print('DOWN 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = 'http://wx.xtuis.cn/XfhmghWTRzitW6RHalZc8AzN5.send?text=' + msg
        requests.get(msgURL)
        playalarm()
    df_hs300['up'] = df_hs300.apply(lambda x: x.close if x.up==True else np.nan, axis=1)
    df_hs300['dw'] = df_hs300.apply(lambda x: x.close if x.dw==True else np.nan, axis=1)

    if len(df_hs300)<240:
        df_hs300 = pd.concat([df_hs300, pd.DataFrame([[]]*(240-len(df_hs300)))])
    df_hs300.reset_index(drop=True,inplace=True)
    df_hs300.reset_index(inplace=True)

    #################

    df_dp['chighlow30'] = df_dp['close'].rolling(30).max()/df_dp['close'].rolling(30).min()*100-100
    df_dp['chighlow60'] = df_dp['close'].rolling(60).max()/df_dp['close'].rolling(60).min()*100-100
    df_dp['cp30'] = (df_dp['close']-df_dp['close'].rolling(30).min())/(df_dp['close'].rolling(30).max()-df_dp['close'].rolling(30).min())
    df_dp['cp60'] = (df_dp['close']-df_dp['close'].rolling(60).min())/(df_dp['close'].rolling(60).max()-df_dp['close'].rolling(60).min())
    df_dp['cm10'] = df_dp['close'].rolling(10).mean()
    df_dp['cabovem10'] = df_dp['close']> df_dp['cm10']

    df_dp.loc[(df_dp['cp30']<0.3) & (df_dp['cp60']<0.3) &  (df_dp['cabovem10']==True) & \
              (df_dp['close']>df_dp['close'].shift(1)), 'up'] = True
    df_dp.loc[(df_dp['cp30']>0.7) & (df_dp['cp60']>0.7) &  (df_dp['cabovem10']==False) & \
              (df_dp['close']<df_dp['close'].shift(1)), 'dw'] = True
    if df_dp['up'].values[-1]==True:
        print('UP 999999 沪市大盘' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_dp['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        playalarm()
    if df_dp['dw'].values[-1]==True:
        print('DOWN 999999 沪市大盘' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_dp['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        playalarm()
    df_dp['up'] = df_dp.apply(lambda x: x.close if x.up==True else np.nan, axis=1)
    df_dp['dw'] = df_dp.apply(lambda x: x.close if x.dw==True else np.nan, axis=1)
    if len(df_dp)<240:
        df_dp = pd.concat([df_dp, pd.DataFrame([[]]*(240-len(df_dp)))])
    df_dp.reset_index(drop=True,inplace=True)
    df_dp.reset_index(inplace=True)

    #################

    # fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,6), dpi=100)
    ax1.set_xticks(np.arange(0, 241, 30))
    ax1.set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330','1400','1430','1500'))
    ax2.set_xticks(np.arange(0, 241, 30))
    ax2.set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330','1400','1430','1500'))
    # ax1.xticks(np.arange(0, 241, 30), ('930', '1000', '1030', '1100', '1130', '1330','1400','1430','1500'))
    # ax2.xticks(np.arange(0, 241, 30), ('930', '1000', '1030', '1100', '1130', '1330','1400','1430','1500'))
    ax1b = ax1.twinx()
    ax1c = ax1.twinx()
    ax1d = ax1.twinx()

    ax2b = ax2.twinx()
    ax2c = ax2.twinx()

    df_dp.plot(x='index', y='close', label='dp', linewidth=1, color='red', ax=ax1,zorder=10)
    df_dp.plot(x='index', y='avg', linewidth=1, markersize=10, color='violet', label='avg', ax=ax1,zorder=11)
    ax1.scatter(df_dp.index, df_dp['up'], marker='^', s=100, c='red',alpha=0.7)
    ax1.scatter(df_dp.index, df_dp['dw'], marker='v',s=100, c='green',alpha=0.7)
    # df_dp.plot(x='index', y='extremaup', marker='*', markersize=8, color='red', label='up',alpha=0.5, ax=ax1,)
    # df_dp.plot(x='index', y='extremadw', marker='*', markersize=8, color='green', label='dw', alpha=0.5,ax=ax1)
    ax1.hlines(y=dp_preclose, xmin=0, xmax=240-3, colors='aqua', linestyles='-', lw=2, label='preclose')

    ax1b.bar(df_dp.index, df_dp.net, label='dpzjlx', color='blue', alpha=0.2, zorder=-15)
    ax1c.bar(df_dp.index, df_dp.northdelta, label='north', color='grey', alpha=0.5, zorder=-14)
    ax1b.plot(df_dp.index, df_dp.bossma5, label='zjlxma5', color='blue', lw=0.5)
    ax1c.plot(df_dp.index, df_dp.northdeltam5, label='northdeltam5', color='black', lw=0.4)
    ax1c.hlines(y=0, xmin=0, xmax=240-3, colors='black', linestyles='-', lw=0.3)
    ax1d.plot(df_dp.index, df_dp.amttrend, label='amttrend', color='green', lw=1.5, alpha=0.5)
    ax1.text(0.5,0.92,' 大盘主力资金(蓝):' + str(round(dp_boss,2)) + ' 北向流入(灰):' + str(round(dp_north,2)) + '  成交量(绿)' ,
             horizontalalignment='center',transform=ax1.transAxes, fontsize=12, fontweight='bold', color='black')

    df_hs300.plot(x='index', y='close', label='hs300', linewidth=1, color='red', ax=ax2,zorder=10)
    df_hs300.plot(x='index', y='avg', linewidth=1, markersize=10, color='violet', label='avg', ax=ax2,zorder=11)
    ax2.hlines(y=hs_preclose, xmin=0, xmax=240-3, colors='aqua', linestyles='-', lw=2, label='preclose')
    ax2.scatter(df_hs300.index, df_hs300['up'], marker='^', s=100, c='red',alpha=0.7)
    ax2.scatter(df_hs300.index, df_hs300['dw'], marker='v',s=100, c='green',alpha=0.7)
    ax2b.bar(df_hs300.index, df_hs300.net, label='zjlx', color='blue', alpha=0.2, zorder=-15)
    ax2b.plot(df_hs300.index, df_hs300.bossma5, label='zjlxm5', color='blue', lw=0.5)
    ax2c.plot(df_hs300.index, df_hs300.upcnt, label='upcnt', color='green', lw=1.5,alpha=0.5 )
    ax2.text(0.5,0.9,'HS300 主力流入:' + str(round(hs300_boss,2)) + ' 市场上涨家数: '+ str(round(upcnt,0)),
             horizontalalignment='center',transform=ax2.transAxes, fontsize=12, fontweight='bold', color='black')
    ax1b.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax1d.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax2c.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)

    ax1.minorticks_on()
    ax1.grid(b=True, which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    # ax1.grid(b=True, which='minor', axis="both", color='k', linestyle='dotted', linewidth=0.5)
    ax1.set(xlabel=None)

    ax2.minorticks_on()
    ax2.grid(b=True, which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    # ax2.grid(b=True, which='minor', axis="both", color='k', linestyle='dotted', linewidth=0.5)
    ax2.set(xlabel=None)

    plt.suptitle('DP HS300 - 时间戳 ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
    plt.tight_layout()
    plt.savefig('static/image.png')
    fig.clf()
    plt.close(fig)
    return

def main():

    global api, Exapi, AmtFactor, dayr1

    try:
        api.close()
        Exapi.close()
    except:
        time.sleep(10)

    try:
        api = TdxHq_API(heartbeat=True)
        Exapi = TdxExHq_API(heartbeat=True)

        if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport )==False:
            print('connection to TDX server not available')
        if TestConnection(Exapi, 'ExHQ', conf.ExHQsvr, conf.ExHQsvrport )==False:
            print('connection to Ex TDX server not available')

        AmtFactor = calAmtFactor(5)
        plotAllzjlx()

        while (time.strftime("%H%M", time.localtime())>='0930' and time.strftime("%H%M", time.localtime())<='1502'):
            if (time.strftime("%H%M", time.localtime())>'1130' and time.strftime("%H%M", time.localtime())<'1300'):
                print('sleep 60s')
                time.sleep(60)
            else:
                plotAllzjlx()
                time.sleep(30)

        api.close()
        Exapi.close()
        return
    except Exception as e: # work on python 3.x
        print('exception msg: '+ str(e))
        print(' *****  exception, restart main ***** ')
        time.sleep(5)
        main()

def image_generation_thread():
     global generation_thread
     main()
     generation_thread.join()


def generate_image():
    image_data = [random.randint(0, 255) for _ in range(256 * 256 * 3)]
    image = Image.frombytes('RGB', (256, 256), bytes(image_data))
    image.save('static/image.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-generation', methods=['POST'])
def start_generation():
    global generation_thread, generation_running

    if not generation_running:
        generation_thread = threading.Thread(target=image_generation_thread)
        generation_thread.start()
        generation_running = True
        print('Image generation stopped.')
        return jsonify({'message': 'Image generation started.'})

    return jsonify({'message': 'Image generation is already running.'})

@app.route('/stop-generation', methods=['POST'])
def stop_generation():
    global generation_thread, generation_running

    if generation_running:
        generation_running = False
        generation_thread.join()
        print('Image generation stopped.')
        return jsonify({'message': 'Image generation stopped.'})

    return jsonify({'message': 'Image generation is not running.'})

if __name__ == '__main__':

    generation_thread = threading.Thread(target=image_generation_thread)
    generation_thread.start()
    generation_running = True

    app.run()

