import baostock as bs
import json, datetime, os, sys
import talib as ta
import numpy as np
import warnings
import time, math, requests
import pandas as pd
import concurrent.futures

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
import utils.tdxExhq_config as conf
from pytdx.reader import TdxDailyBarReader,TdxExHqDailyBarReader,BlockReader

# from flaml import AutoML
# from supervised.automl import AutoML # mljar-supervised
import matplotlib.pyplot as plt
import pathlib

import seaborn as sns
from sklearn.svm import SVR
from scipy.stats import randint
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import validation_curve

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense,Dropout

import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
try:
    from sklearn.preprocessing import Imputer
except:
    from sklearn.impute import  SimpleImputer as Imputer

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
#  switch to backends that do not use GUI: Agg, Cairo, PS, PDF, or SVG.
#  https://matplotlib.org/stable/faq/howto_faq.html#work-with-threads
plt.switch_backend('agg')
# sns.color_palette("Reds")
# sns.set_style("whitegrid")
# sns.set_style({'font.sans-serif':['SimHei','Arial']})

# pd.describe_option()
pd.options.mode.chained_assignment = None
pd.set_option('precision', 6)
pd.set_option('display.precision', 4)
pd.set_option('display.float_format',  '{:.3f}'.format)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('expand_frame_repr', False)
pd.set_option('use_inf_as_na',True)
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
np.random.seed(42)

from sklearn.model_selection import train_test_split  # 切分训练集与测试集
from sklearn.preprocessing import StandardScaler  # 标准化数据
from sklearn.preprocessing import LabelEncoder   # 标签化分类变量

stkrate = 0.0014
etfrate = 0.0006

output_fn_prefix = "D:\\stockstudy\\win2021\\daily_output\\"
tdx_root_path = "D:\\zszq\\"
bao_data_path = "D:\\stockstudy\\win2021\\baodata\\"
embk_data_path = "D:\\stockstudy\\win2021\\EM_BK_data\\"

pullrealtime_start = "0930"
pullrealtime_end = "2200"

stkFeatures = True

##################### common utilities #####################

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return (self)

    def transform(self, X):
        '''X is a DataFrame'''
        return (X[self.attribute_names].values)

def BuildModel():
    model = Sequential()
    model.add(Dense(128, input_dim=46, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# 计算复权价格
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

def bkFromDat(fn):
    # 读取板块信息
    # 参数： 板块文件名称，可以取的值限于

    df = BlockReader().get_df(fn)
    return df

    # # 板块相关参数
    # BLOCK_SZ = "block_zs.dat"
    # BLOCK_FG = "block_fg.dat"
    # BLOCK_GN = "block_gn.dat"
    # BLOCK_DEFAULT = "block.dat"
    # api.get_and_parse_block_info("block.dat")
    # # 或者用我们定义好的params
    # api.get_and_parse_block_info(TDXParams.BLOCK_SZ)

def GetTdxBKlist(tdxzs_fn, tdxstkmap_fn):

    if os.path.exists(tdxzs_fn) and os.path.exists(tdxstkmap_fn):
        print('\nget bk info from files \n', tdxzs_fn, '\n', tdxstkmap_fn)
        df_zs = pd.read_csv(tdxzs_fn, encoding='gbk', dtype=str)
        df_zs = df_zs[(df_zs['bkclass'] == '2') | (df_zs['bkclass'] == '4')]  # 350 rows
        #  "D:\\stockstudy\\win2021\\basepy\\TdxBKml\\tdx_zs_all_20220216.csv"
        # bkname	bkcode	bkclass	tdxbkcode
        # 有色	    880324	2	    T0202
        df_bk_stk_map = pd.read_csv(tdxstkmap_fn, encoding='gbk', dtype=str)
        # "D:\\stockstudy\\win2021\\basepy\\TdxBKml\\stk_hy_gn_mapping_all_20220216.csv"
        # bkname	bkcode	stkcode
        # 成渝特区	880522	601208
        return df_zs, df_bk_stk_map
    else:
        print("can not find ", tdxzs_fn, tdxstkmap_fn)

    print('\nReading Tdx BK info ... ')

    # # fn_blk = 'D:\\zszq\\T0002\\hq_cache\\block.dat'        # 一般板块
    # # fn_fg = 'D:\\zszq\\T0002\\hq_cache\\block_fg.dat'        # 一般板块
    # # fn_zs = 'D:\\zszq\\T0002\\hq_cache\\block_zs.dat'        # 一般板块
    # # T0002\hq_cache \block_gn.dat    概念板块
    # # T0002\hq_cache\block_fg.dat     风格板块
    # # T0002\hq_cache\block_zs.dat     指数板块
    # # df_blk = bkFromDat(fn_blk)
    # # df_fg = bkFromDat(fn_fg)
    # # df_zs = bkFromDat(fn_zs)
    fn_gn = 'D:\\zszq\\T0002\\hq_cache\\block_gn.dat'  # 一般板块
    fn_hy = 'D:\\zszq\\T0002\\hq_cache\\tdxhy.cfg'  # 个股所属行业
    fn_zs = 'D:\\zszq\\T0002\\hq_cache\\tdxzs.cfg'  # 指数列表
    df_gn = bkFromDat(fn_gn)
    # df_gn.to_csv("stk_gn_mapping_all.csv", encoding='gbk', index=False)
    df_stk_hy, df_zs = text2df(fn_hy, fn_zs)
    df_zs.to_csv(tdxzs_fn, encoding='gbk', index=False)
    df_zs = df_zs[(df_zs['bkclass'] == '2') | (df_zs['bkclass'] == '4')]  # 350 rows

    df_zs_gn = df_zs[df_zs['bkclass'] == '4'][['bkname', 'bkcode']]  # bkname	bkcode	bkclass	tdxbkcode
    # df_stk_hy.to_csv("stk_hy_mapping_all.csv", encoding='gbk', index=False)
    df_map_gn = pd.merge(df_zs_gn, df_gn, left_on='bkname', right_on='blockname', how='right')
    df_map_gn.dropna(axis=0, how='any', inplace=True)
    df_map_gn = df_map_gn[['bkname', 'bkcode', 'code']]
    df_map_gn.columns = ['bkname', 'bkcode', 'stkcode']
    df_stk_hy = df_stk_hy[['bkname', 'bkcode', 'stkcode']]
    df_BK_all = pd.concat([df_map_gn, df_stk_hy])   # 行业指数+概念指数映射到个股
    # df_BK_all.to_csv("stk_hy_gn_mapping_all.csv", encoding='gbk', index=False)
    df_BK_all.to_csv(tdxstkmap_fn, encoding='gbk', index=False)

    return df_zs, df_BK_all

def getEMBKList():
    # url = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112306975119471497313_1641425906394&pn=1&pz=500&po=1&np=1&fields=f12%2Cf13%2Cf14%2Cf62&fid=f62&fs=m%3A90%2Bt%3A3&ut=b2884a393a59ad64002292a3e90d46a5&_=1641425906395'
    # updated on 20220315
    url = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112307630547660526908_1647313958732&pn=1&pz=500&po=1&np=1&fields=f12%2Cf13%2Cf14%2Cf62&fid=f62&fs=m%3A90%2Bt%3A3&ut=b2884a393a59ad64002292a3e90d46a5&_=1647313958733'
    res = requests.get(url)
    try:
        # data1 = json.loads(res.text[42:-2])['data']['diff']
        data1 = json.loads(res.text[42:-2])['data']['diff']
    except:
        return pd.DataFrame()
    min = pd.DataFrame(data1)
    min.columns = ['code', 'marketcode', 'name', 'flow']
    return min

def day2quarter(datestr):

    hr, min,sec = map(int, timestr.split(':'))
    tradingmin = hr*60+min - 9*60-30
    if tradingmin>0 and tradingmin<120:
        return factor[tradingmin-1]
    elif tradingmin>=120 and tradingmin<210:
        return factor[119]
    elif tradingmin>=210 and tradingmin<330:
        return factor[tradingmin-91]
    else:
        return 1

def caclDates4ml(ldays, tdays, vdays, pdays):
    # leadingdays = 100
    # traindays = 3000
    # validdays = 600
    # predictdays = 1

    dayofweek = datetime.datetime.now().weekday()  # mon-fri :  0-4
    today_cal = datetime.datetime.now().strftime('%Y-%m-%d')
    today_stk = today_cal if dayofweek < 5 else (
                datetime.datetime.now() - datetime.timedelta(days=dayofweek - 4)).strftime("%Y-%m-%d")

    basedate = datetime.datetime.strptime(today_stk, '%Y-%m-%d') - datetime.timedelta(days=pdays - 1)
    pull_s = (basedate - datetime.timedelta(days= ldays + tdays + vdays + pdays)).strftime("%Y-%m-%d")
    pull_e = datetime.datetime.now().strftime('%Y-%m-%d')
    train_s = (basedate - datetime.timedelta(days= tdays + vdays + pdays)).strftime("%Y-%m-%d")
    train_e = (basedate - datetime.timedelta(days=vdays + pdays)).strftime("%Y-%m-%d")  # basedate.strftime("%Y-%m-%d")
    valid_s = (basedate - datetime.timedelta(days=vdays)).strftime("%Y-%m-%d")
    valid_e = datetime.datetime.now().strftime('%Y-%m-%d')

    return pull_s,pull_e,train_s,train_e,valid_s,valid_e, today_stk

def week_day(str):
    return datetime.date.weekday(datetime.datetime.strptime(str, "%Y-%m-%d"))

def mybin(val):
    if val < 0.34:
        return 1
    elif val < 0.67:
        return 2
    else:
        return 3

def myFixBin(val, cnt=5, minv=0, maxv=0):
    result = []
    if minv==maxv:
        vmax = val.dropna().max()
        vmin = val.dropna().min()
        # vgap = vmax-vmin
    else:
        vmax = max(minv,maxv)
        vmin = min(minv,maxv)
    vgap = vmax - vmin
    if vgap==0:
        return [5 for i in val]
    else:
        result = [math.ceil((i-vmin)/vgap*cnt) if np.isnan(i)==False else np.nan for i in val]
        return [1 if i==0 else i for i in result]

def mybin5(val):
    if val < 0.2:
        return 1
    elif val < 0.4:
        return 2
    elif val < 0.6:
        return 3
    elif val < 0.8:
        return 4
    else:
        return 5

def myCeil(data):
    return [math.ceil(i) for i in data]

def computeKDJ(df):
    low_list = df['low'].rolling(window=9).min()
    high_list = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['kdj'] = [1 if (x >= 0 and y < 0) else 0 for x, y in zip(list(df['J']), [1] + list(df['J'])[:-1])]
    df.drop(['K', 'D', 'J'], axis=1, inplace=True)
    return (df['kdj'])

def computeKDJv2(df):
    low_list = df['low'].rolling(window=9).min()
    high_list = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    return rsv

def computeRSI(df):
    rsi = ta.RSI(df['close'], timeperiod=14)
    df["rsi"] = [1 if i < 30 else 0 for i in rsi]
    return (df["rsi"])

def computeRSIv2(df):
    rsi = ta.RSI(df['close'], timeperiod=14)
    # df["rsi"] = [1 if i < 30 else 0 for i in rsi]
    return (rsi)

def computeOBV(df):
    df["ob"] = ta.OBV(df['close'].values, df['volume'].values)
    obv = df["ob"].rolling(20).mean()
    df['obv'] = [1 if (o > o1 and c <= c1) else 0 for o, o1, c, c1 in zip(list(obv), [1] + list(obv)[:-1],
                                                                          list(df['close']),
                                                                          [1] + list(df['close'])[:-1])]
    df.drop(['ob'], axis=1, inplace=True)
    return (df["obv"])

def computeBoll(df):
    upper, middle, lower = ta.BBANDS(df["close"], timeperiod=20, matype=ta.MA_Type.SMA)
    df["lower"] = lower
    df['boll'] = [1 if (c1 < l1 and c > l) else 0 for l, l1, c, c1 in
                  zip(list(df['lower']), [1] + list(df['lower'])[:-1],
                      list(df['close']), [1000] + list(df['close'])[:-1])]
    df.drop(['lower'], axis=1, inplace=True)
    return (df['boll'])

def computeBollv2(df):
    upper, middle, lower = ta.BBANDS(df["close"], timeperiod=20, matype=ta.MA_Type.SMA)
    df["bwidth"] = (upper/lower-1)*100
    df['bupperdelta'] = (upper/upper.shift(1)-1)*100
    return (df['bwidth'], df['bupperdelta'])

def computeBias(input,n):
    df = input.copy()
    df.columns=['close']
    df['m1'] = df['close'].rolling(n, min_periods=1).mean()
    df['m2'] = df['close'].rolling(2*n, min_periods=1).mean()
    df['m3'] = df['close'].rolling(4*n, min_periods=1).mean()
    df['b1'] = df.apply(lambda x: (x.close-x.m1)/x.m1*100,axis=1)
    df['b2'] = df.apply(lambda x: (x.close-x.m2)/x.m2*100,axis=1)
    df['b3'] = df.apply(lambda x: (x.close-x.m3)/x.m3*100,axis=1)
    df['b0'] = df.apply(lambda x: -(x.b1+2*x.b2+3*x.b3)/6,axis=1)
    df['bias'] = df['b0'].rolling(3, min_periods=1).mean()
    # df['biasup'] = df['bias'] < df['bias'].shift(1)
    # df.drop(columns=['m1','m2','m3','b1','b2','b3','b0'],inplace=True)
    return df['bias']   #,df['biasup']

def calcIndicator(m5,m10,m20):
    if m5>m10 and m10>m20:
        return 1
    if m5<m10 and m10<m20:
        return -1
    else:
        return 0

def custom_resampler(arraylike):
    return np.sum(arraylike) + 5

def calSafeLeft(df):
    df_sl = [0] * 50
    df.index = pd.to_datetime(df['date'])
    for i in range(50,len(df)):
        temp = df[i-50:i+1]
        if list(temp['date'])[-1] == '2020-07-28':
            pass
        if list(temp['kline'])[-1]==1 and list(temp['kline'])[-2]==0 and list(temp['abovema5'])[-2]==0:
            temp_wk = temp.resample('W').ffill()
            temp_wk['ma5'] = temp_wk['close'].rolling(5).mean()
            temp_wk['abovema5'] = temp_wk.apply(lambda x: 1 if x.close>x.ma5 else 0, axis=1)
            if list(temp_wk['abovema5'])[-3:] == [1,1,1]:
                df_sl.append(1)
            else:
                df_sl.append(0)
        else:
            df_sl.append(0)
            continue
    return df_sl

def min2day(timestr):
    factor = [30.7395,20.5736,15.8613,13.0092,11.1578,9.8177,8.8125,8.0707,7.4616,6.9545,6.4752,6.0869,
              5.7620,5.4838,5.2396,5.0007,4.7964,4.6182,4.4579,4.3133,4.1607,4.0341,3.9201,3.8174,
              3.7208,3.6250,3.5401,3.4608,3.3838,3.3129,3.2326,3.1652,3.1035,3.0456,2.9930,2.9433,
              2.8948,2.8507,2.8081,2.7674,2.7227,2.6848,2.6485,2.6142,2.5811,2.5484,2.5176,2.4888,
              2.4612,2.4350,2.4049,2.3779,2.3535,2.3304,2.3081,2.2867,2.2657,2.2459,2.2266,2.2078,
              2.1842,2.1634,2.1444,2.1266,2.1095,2.0926,2.0763,2.0606,2.0453,2.0305,2.0136,1.9986,
              1.9844,1.9709,1.9579,1.9447,1.9320,1.9200,1.9083,1.8966,1.8839,1.8728,1.8619,1.8512,
              1.8409,1.8307,1.8206,1.8108,1.8013,1.7921,1.7813,1.7713,1.7617,1.7523,1.7435,1.7346,
              1.7260,1.7179,1.7099,1.7021,1.6932,1.6849,1.6772,1.6699,1.6626,1.6550,1.6477,1.6403,
              1.6332,1.6264,1.6189,1.6119,1.6053,1.5991,1.5929,1.5870,1.5811,1.5756,1.5701,1.5641,
              1.5484,1.5415,1.5353,1.5292,1.5231,1.5168,1.5105,1.5047,1.4989,1.4931,1.4864,1.4803,
              1.4743,1.4684,1.4629,1.4574,1.4518,1.4464,1.4408,1.4351,1.4286,1.4228,1.4172,1.4117,
              1.4062,1.4007,1.3953,1.3897,1.3841,1.3785,1.3718,1.3664,1.3612,1.3561,1.3511,1.3464,
              1.3416,1.3369,1.3323,1.3277,1.3227,1.3179,1.3133,1.3088,1.3044,1.2999,1.2955,1.2911,
              1.2868,1.2826,1.2779,1.2735,1.2694,1.2653,1.2613,1.2571,1.2532,1.2494,1.2456,1.2418,
              1.2371,1.2328,1.2289,1.2251,1.2214,1.2176,1.2137,1.2101,1.2065,1.2029,1.1990,1.1954,
              1.1918,1.1883,1.1848,1.1812,1.1777,1.1743,1.1708,1.1674,1.1636,1.1602,1.1567,1.1533,
              1.1498,1.1463,1.1429,1.1396,1.1362,1.1329,1.1287,1.1248,1.1211,1.1173,1.1136,1.1099,
              1.1062,1.1024,1.0987,1.0949,1.0907,1.0866,1.0826,1.0786,1.0745,1.0701,1.0659,1.0615,
              1.0572,1.0527,1.0475,1.0425,1.0370,1.0313,1.0257,1.0192,1.0124,1.0118,1.0118,1.0]
    hr, min,sec = map(int, timestr.split(':'))
    tradingmin = hr*60+min - 9*60-30
    if tradingmin>0 and tradingmin<120:
        return factor[tradingmin-1]
    elif tradingmin>=120 and tradingmin<210:
        return factor[119]
    elif tradingmin>=210 and tradingmin<330:
        return factor[tradingmin-91]
    else:
        return 1

def roundTo3(y):
    return [str(round(x, 4)) for x in y]

def text2df(hyfn, zsfn):

    df_zs = pd.read_csv(zsfn, sep='|', encoding='gbk', dtype=str, header=None)
    # 酿酒|880380|2|1|0|T0305
    # 2-通达信行业板块 3-地区板块 4-概念板块 5-风格板块 8-申万行业，  only 2 & 4 are used.
    df_zs.columns = ['bkname', 'bkcode', 'bkclass', 'f3','f4','tdxbkcode']
    df_zs = df_zs[['bkname', 'bkcode', 'bkclass', 'tdxbkcode']]

    df_hy = pd.read_csv(hyfn, sep='|',dtype=str, header=None)
    # 0|000001|T1001|||X500102
    df_hy.columns = ['mkt', 'stkcode', 'tdxbkcode','f3','f4','f5']
    df_hy = df_hy[['mkt', 'stkcode', 'tdxbkcode']]
    df_hy.dropna(axis=0, how='any', inplace=True)
    df_hy.to_csv("stk_hy_orig_all.csv",encoding='gbk',index=False)
    df_stk_hy = pd.merge(df_hy, df_zs, on='tdxbkcode', how='left')

    return df_stk_hy, df_zs

##################### data pulling #####################

def prepDapan(code, name, df_status, freq):
    global today, stocklist

    # df_status = df[df['code']==code]
    if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
        and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:
        df_status = df_status[df_status['date'] != datetime.datetime.now().strftime('%Y-%m-%d')]
        newdata = pullRealtimeDataFromEMweb(code)
        if len(newdata) == 1:
            print("+", end='', flush=False)
            # df_status = df_status.append(newdata[0], ignore_index=True)
            df_status = df_status.append({
                'date': newdata[0]['date'], 'code': newdata[0]['code'],
                'open': newdata[0]['open'], 'high': newdata[0]['high'],
                'low': newdata[0]['low'], 'close': newdata[0]['close'],
                'volume': newdata[0]['volume'], 'amount': newdata[0]['amount'],
                'turn': newdata[0]['low'], 'pctChg': newdata[0]['pctChg']
            }, ignore_index=True)
        else:
            return pd.DataFrame()

    # add more features with calculation
    df_status['code'] = code
    df_status['sname'] = name

    try:
        if df_status.date[len(df_status) - 1] == datetime.datetime.now().strftime('%Y-%m-%d'):
            df_status.turn[len(df_status) - 1] = df_status.volume[len(df_status) - 1] / df_status.volume[len(df_status) - 2] * \
                                         df_status.turn[len(df_status) - 2]
    except:
        pass

    if freq == 'Q':
        df_status.index = pd.to_datetime(df_status['date'])
        df_q = pd.DataFrame()
        df_q['open'] = df_status.open.resample('Q').apply(lambda x: x[0] if len(x) > 0 else 0)
        df_q['high'] = df_status.high.resample('Q').apply(lambda x: x.max() if len(x) > 0 else 0)
        df_q['low'] = df_status.low.resample('Q').apply(lambda x: x.min() if len(x) > 0 else 0)
        df_q['close'] = df_status.close.resample('Q').apply(lambda x: x[-1] if len(x) > 0 else 0)

        qleftdays = (df_q.index[-1] - df_status.index[-1]).days
        df_q['volume'] = df_status.volume.resample('Q').sum() / 100000000
        df_q['amount'] = df_status.amount.resample('Q').sum() / 100000000
        df_q.volume[-1] = df_q.volume[-1] / max((90 - qleftdays), 1) * 90
        df_q.amount[-1] = df_q.amount[-1] / max((90 - qleftdays), 1) * 90

        df_q['date'] = df_q.index.strftime('%Y-%m-%d')
        df_q['pctChg'] = (df_q["close"] / df_q["close"].shift(1) - 1) * 100
        df_q['sname'] = name
        df_q['code'] = code

        return df_q[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'sname']]

    elif freq == "M":
        df_status.index = pd.to_datetime(df_status['date'])
        df_q = pd.DataFrame()
        df_q['open'] = df_status.open.resample('M').apply(lambda x: x[0] if len(x) > 0 else 0)
        df_q['high'] = df_status.high.resample('M').apply(lambda x: x.max() if len(x) > 0 else 0)
        df_q['low'] = df_status.low.resample('M').apply(lambda x: x.min() if len(x) > 0 else 0)
        df_q['close'] = df_status.close.resample('M').apply(lambda x: x[-1] if len(x) > 0 else 0)

        qleftdays = (df_q.index[-1] - df_status.index[-1]).days
        df_q['volume'] = df_status.volume.resample('M').sum() / 100000000
        df_q['amount'] = df_status.amount.resample('M').sum() / 100000000
        df_q.volume[-1] = df_q.volume[-1] / max((31 - qleftdays),1) * 31
        df_q.amount[-1] = df_q.amount[-1] / max((31 - qleftdays),1) * 31

        df_q['date'] = df_q.index.strftime('%Y-%m-%d')
        df_q['pctChg'] = (df_q["close"] / df_q["close"].shift(1) - 1) * 100
        df_q['sname'] = name
        df_q['code'] = code
        return df_q[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'sname']]

    elif freq == 'W':
        df_status.index = pd.to_datetime(df_status['date'])
        df_q = pd.DataFrame()
        df_q['open'] = df_status.open.resample('W').apply(lambda x: x[0] if len(x) > 0 else 0)
        df_q['high'] = df_status.high.resample('W').apply(lambda x: x.max() if len(x) > 0 else 0)
        df_q['low'] = df_status.low.resample('W').apply(lambda x: x.min() if len(x) > 0 else 0)
        df_q['close'] = df_status.close.resample('W').apply(lambda x: x[-1] if len(x) > 0 else 0)

        df_q = df_q[df_q['close'] > 0]
        qleftdays = df_status.index[-1].weekday()
        df_q['volume'] = df_status.volume.resample('W').sum() / 100000000
        df_q['amount'] = df_status.amount.resample('W').sum() / 100000000
        df_q.volume[-1] = df_q.volume[-1] / (qleftdays + 1) * 5
        df_q.amount[-1] = df_q.amount[-1] / (qleftdays + 1) * 5

        df_q['date'] = df_q.index.strftime('%Y-%m-%d')
        df_q['pctChg'] = (df_q["close"] / df_q["close"].shift(1) - 1) * 100
        df_q['sname'] = name
        df_q['code'] = code
        return df_q[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'sname']]

    elif freq == 'D':
        return df_status[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'sname']]

    else:
        print('freq ', freq, 'error.  it should be Q/M/W/D')
        return pd.DataFrame()

def prepDapanAllPeriods(code, df_status):

    if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
        and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:
        df_status = df_status[df_status['date'] != datetime.datetime.now().strftime('%Y-%m-%d')]
        newdata = pullRealtimeDataFromEMweb(code)
        if len(newdata) == 1:
            print("+", end='', flush=False)
            # df_status = df_status.append(newdata[0], ignore_index=True)
            df_status = df_status.append({
                'date': newdata[0]['date'], 'code': newdata[0]['code'],
                'open': newdata[0]['open'], 'high': newdata[0]['high'],
                'low': newdata[0]['low'], 'close': newdata[0]['close'],
                'volume': newdata[0]['volume'], 'amount': newdata[0]['amount'],
                'turn': newdata[0]['low'], 'pctChg': newdata[0]['pctChg']
            }, ignore_index=True)
        else:
            return pd.DataFrame()

    dapan_D =  df_status[['date', 'close']]
    df_status = df_status[['date','close']]
    df_status.index = pd.to_datetime(df_status['date'])

    dapan_W = pd.DataFrame()
    dapan_W['close'] = df_status.close.resample('W').apply(lambda x: x[-1] if len(x) > 0 else 0)
    dapan_W['date'] = dapan_W.index.strftime('%Y-%m-%d')
    dapan_W = dapan_W[dapan_W['close']>0]
    dapan_W.reset_index(inplace=True, drop=True)

    dapan_M = pd.DataFrame()
    dapan_M['close'] = df_status.close.resample('M').apply(lambda x: x[-1] if len(x) > 0 else 0)
    dapan_M['date'] = dapan_M.index.strftime('%Y-%m-%d')
    dapan_M.reset_index(inplace=True, drop=True)

    dapan_Q = pd.DataFrame()
    dapan_Q['close'] = df_status.close.resample('Q').apply(lambda x: x[-1] if len(x) > 0 else 0)
    dapan_Q['date'] = dapan_Q.index.strftime('%Y-%m-%d')
    dapan_Q.reset_index(inplace=True, drop=True)

    return dapan_D, dapan_W, dapan_M, dapan_Q

def readDapanData(startdate, enddate, baopath, freq):

    # dpcode = ['sh.000001', 'sz.399001']
    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}
    # for code in dpcode:
    df_sh = pd.read_csv(baopath + 'bao-sh.000001.csv',dtype=dtype_dic, encoding='gbk', )
    # df_sh = df_sh[(df_sh['df_sh']>=startdate) & (df_sh['date']<=enddate)]
    df_sz = pd.read_csv(baopath + 'bao-sz.399001.csv',dtype=dtype_dic, encoding='gbk', )
    # df_sz = df_sz[(df_sz['df_sh']>=startdate) & (df_sz['date']<=enddate)]

    df_sh = prepDapan('sh.000001', 'shdp', df_sh, freq)
    df_sh.columns = ['date', 'shcode', 'shopen', 'shhigh', 'shlow', 'shclose', 'vol_sh', 'amt_sh', 'shpctChg','shname']
    df_sz = prepDapan('sz.399001', 'szdp', df_sz, freq)
    df_sz.columns = ['date', 'szcode', 'szopen', 'szhigh', 'szlow', 'szclose', 'vol_sz', 'amt_sz', 'szpctChg','szname']

    df_sh.reset_index(inplace=True, drop=True)
    df_sz.reset_index(inplace=True, drop=True)

    df_dp = pd.merge(left=df_sh[['date','shclose','vol_sh', 'amt_sh']],
                     right=df_sz[['date','szclose','vol_sz', 'amt_sz']],how='left',on='date')
    df_dp['vol'] = df_dp.vol_sh + df_dp.vol_sz
    df_dp['amt'] = df_dp.amt_sh + df_dp.amt_sz
    df_dp['dpamtrk'] = df_dp['amt'].rolling(40+1).apply(lambda x: pd.Series(x).rank().iloc[-1])
    df_dp['dpamtrank'] = 2*(df_dp['dpamtrk']-40-1)/40+1 # rank_days = 40

    df_dp['r10sh'] = df_dp['shclose']/df_dp['shclose'].shift(10)-1
    df_dp['r10sz'] = df_dp['szclose']/df_dp['szclose'].shift(10)-1
    df_dp['pctR10'] = df_dp['r10sh'] + df_dp['r10sz']

    df_dp['biassh'] = computeBias(df_dp[['shclose']],10)
    df_dp['biassz'] = computeBias(df_dp[['szclose']],10)
    df_dp['dpbias'] = df_dp.biassh + df_dp.biassz
    df_dp['dpbiasup'] = df_dp.dpbias < df_dp.dpbias.shift(1)

    df_dp['dpenv'] = df_dp['shclose']/df_dp['shclose'].shift(10)/2 + \
                      df_dp['szclose']/df_dp['szclose'].shift(10)/2 -1
    # df_dp = df_dp[(df_dp['date']>=startdate) & (df_dp['date']<=enddate)]
    df_dp = df_dp[(df_dp['date']>=startdate)]

    df_dp.loc[df_dp.pctR10>=0.04, 'r10env'] = 1
    df_dp.loc[df_dp.pctR10<=-0.04, 'r10env'] = -1
    df_dp.loc[(df_dp.pctR10<0.04) & (df_dp.pctR10>-0.04), 'r10env'] = 0

    print(df_dp[['date','amt','dpamtrank','dpenv','r10env','dpbias','dpbiasup','szclose', 'shclose']].tail(5))
    result = df_dp[['date','dpamtrank','dpenv','dpbias','dpbiasup','szclose', 'shclose']]

    return result

def readDapanDataAllPeriods(ds, ws, ms, qs, baopath):

    # dpcode = ['sh.000001', 'sz.399001']
    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}

    df_sh = pd.read_csv(baopath + 'bao-sh.000001.csv',dtype=dtype_dic, encoding='gbk', )
    df_sz = pd.read_csv(baopath + 'bao-sz.399001.csv',dtype=dtype_dic, encoding='gbk', )

    df_shD, df_shW,df_shM,df_shQ = prepDapanAllPeriods('sh.000001', df_sh)
    df_shD.columns = ['date', 'shclose']
    df_shW.columns = ['shclose','date']
    df_shM.columns = ['shclose','date']
    df_shQ.columns = ['shclose','date']
    df_szD, df_szW,df_szM,df_szQ = prepDapanAllPeriods('sz.399001', df_sz)
    df_szD.columns = ['date', 'szclose']
    df_szW.columns = ['szclose','date']
    df_szM.columns = ['szclose','date']
    df_szQ.columns = ['szclose','date']

    df_dpD = pd.merge(left=df_shD[['date','shclose']], right=df_szD[['date','szclose']],how='left',on='date')
    df_dpW = pd.merge(left=df_shW[['date','shclose']], right=df_szW[['date','szclose']],how='left',on='date')
    df_dpM = pd.merge(left=df_shM[['date','shclose']], right=df_szM[['date','szclose']],how='left',on='date')
    df_dpQ = pd.merge(left=df_shQ[['date','shclose']], right=df_szQ[['date','szclose']],how='left',on='date')

    df_dpD = df_dpD[df_dpD['date']>=ds]
    df_dpW = df_dpW[df_dpW['date']>=ws]
    df_dpM = df_dpM[df_dpM['date']>=ms]
    df_dpQ = df_dpQ[df_dpQ['date']>=qs]

    return df_dpD,df_dpW,df_dpM,df_dpQ


# 数据获取一级函数 - TdxBK
def getTdxBKdata(TdxBKlist, data_pull_s, data_pull_e, batchcnt, freq):
    batchCnt = batchcnt
    stockPerBatch = len(TdxBKlist) // batchCnt + 1
    stockBatchList = []
    # bk_black = ['次新股','含可转债','央企改革']
    bk_black = []
    # names = locals()
    for i in range(batchCnt):
        sl_temp = TdxBKlist[i * stockPerBatch: (i + 1) * stockPerBatch]
        stocklist_temp = {}
        for index, row in sl_temp.iterrows():
            if row['bkname'] not in bk_black:
                stocklist_temp[row['bkcode']] = row['bkname']
            else:
                print(row['bkname'], 'skipped')

        stockBatchList.append(stocklist_temp)

    with concurrent.futures.ProcessPoolExecutor() as executor:  # 多进程
        results = [executor.submit(processTdxBKBatches, i, batch, data_pull_s, data_pull_e, freq) \
                   for i, batch in enumerate(stockBatchList)]
        df_concat = pd.concat([result.result() for result in results], ignore_index=True)

    # df_concat = processTdxBKBatches(0, stockBatchList[0], data_pull_s, data_pull_e, freq)

    return df_concat

# 数据获取二级函数 - TdxBK
def processTdxBKBatches(seq, batch, data_pull_s, data_pull_e, freq):
    global conf

    api = TdxHq_API()
    if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport )==False:
        print('connection to TDX server not available')

    print("processing batch ", seq)
    # codelist = batch.keys()

    batch_start = time.time()
    df_batch = pd.DataFrame()
    # dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
    #              'amount':float,'turn':float,'pctChg':float}

    counter = 0
    for key, value in batch.items():
        df_single = getTdxBKDataFromPytdx(key, value, data_pull_s, api, freq)
        df_batch = pd.concat([df_batch, df_single])

        counter += 1
        if counter % 100 == 0:
            print('.' + str(counter))
        else:
            print('.', end='', flush=True)

    df_batch.fillna({'aft1D': 0, 'aft2D': 0,  'aft3D': 0, 'aft5D': 0, 'aft8D': 0, 'aft10D': 0, }, inplace=True)
    print('\nbatch:', seq, 'rows:', len(df_batch), ' Data Prepared in ', round(time.time() - batch_start, 0), ' secs.')

    api.close()
    return df_batch

# 数据获取三级函数 - TdxBK
def getTdxBKDataFromPytdx(code, name, datefrom,api, freq):

    reader = TdxDailyBarReader()

    if len(code)==6 and code[:2]=='88': # 行业指数
        df = reader.get_df("D:\\zszq\\Vipdoc\\sh\\lday\\sh"+code+".day")
        market = 1
    elif code=='000300':
        df = reader.get_df("D:\\zszq\\Vipdoc\\sh\\lday\\sh"+code+".day")
        market = 1
    elif len(code)==8 and code[:2]=='sh':
        df = reader.get_df("D:\\zszq\\Vipdoc\\sh\\lday\\sh"+code[2:]+".day")
        market = 1
        code = code[-6:]
    elif len(code)==8 and code[:2]=='sz':
        df = reader.get_df("D:\\zszq\\Vipdoc\\sz\\lday\\sz"+code[2:]+".day")
        market = 0
        code = code[-6:]
    elif len(code)==8 and code[:4]=='1000': #期权
        df = reader.get_df("D:\\zszq\\Vipdoc\\sh\\lday\\sh"+code+".day")
        market = 1
    else:
        print(code, 'not coded!')

    df['date'] = df.index
    df.reset_index(drop=True,inplace=True)
    df = df[df['date']>=pd.to_datetime(datefrom)]
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strftime(x, "%Y-%m-%d"))

    if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
        and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:
        date, close, amount, vol,high,low,open,preclose = getTdxRealtime(api, market, code)
        df = df[df['date'] < date]
        df = df.append({'close': close, 'high': high, 'low': low, 'open': open,
                        'amount': amount, 'volume': vol, 'date': date},  ignore_index=True)
        print('+', end="")
    if freq == 'Q':
        df_single = prepTdxBKQdata(code, name, df)
        return df_single
    elif freq == "M":
        df_single = prepTdxBKMdata(code, name, df)
        return df_single
    elif freq == 'W':
        df_single = prepTdxBKWdata(code, name, df)
        return df_single
    elif freq == 'D':
        df_single = prepTdxBKDdata(code, name, df)
        return df_single
    else:
        print('freq ', freq, 'error.  it should be Q/M/W/D')
        return pd.DataFrame()

# 数据获取四级函数 - TdxBK
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
        print(market, code, "no data.")

# 数据获取四级函数 - TdxBK
def prepTdxBKQdata(code, name, df_status):
    # global today, stocklist

    if len(df_status)<200:
        return pd.DataFrame()

    df_status['preclose'] = df_status['close'].shift(1)
    df_status['pctChg'] = (df_status['close']/df_status['preclose']-1)*100
    pctindex = df_status.index[abs(df_status['pctChg'])>15].tolist()
    if len(pctindex) > 0:
        df_status = df_status[pctindex[-1]:]
        if len(df_status)<200:
            return pd.DataFrame()

    df_status.index = pd.to_datetime(df_status['date'])

    df_q = pd.DataFrame()

    # df_q['open'] = df_status.open.resample('Q').apply(custom_resampler)
    df_q['open'] = df_status.open.resample('Q').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['high'] = df_status.high.resample('Q').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['low'] = df_status.low.resample('Q').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['close'] = df_status.close.resample('Q').apply(lambda x: x[-1] if len(x)>0 else 0)

    qleftdays = (df_q.index[-1] - df_status.index[-1]).days
    # df_q['turn'] = df_status.turn.resample('Q').sum()
    # df_q.turn[-1] = df_q.turn[-1] / (90 - qleftdays) * 90
    df_q['volume'] = df_status.volume.resample('Q').sum()/100000000
    df_q['amount'] = df_status.amount.resample('Q').sum()/100000000

    df_q.volume[-1] = df_q.volume[-1] / max((90 - qleftdays),1) * 90
    df_q.amount[-1] = df_q.amount[-1] / max((90 - qleftdays),1) * 90

    df_q['date'] = df_q.index.strftime('%Y-%m-%d')
    df_q['preclose'] = df_q['close'].shift(1)

    df_q['opct'] = (df_q['open'] / df_q['preclose'] - 1) * 100
    df_q['hpct'] = (df_q['high'] / df_q['preclose'] - 1) * 100
    df_q['lpct'] = (df_q['low'] / df_q['preclose'] - 1) * 100

    df_q['pctChg'] = (df_q["close"]/df_q["close"].shift(1)-1)*100
    df_q['pctR2'] = (df_q["close"]/df_q["close"].shift(2)-1)*100
    df_q['pctR3'] = (df_q["close"]/df_q["close"].shift(3)-1)*100
    df_q['pctR4'] = (df_q["close"]/df_q["close"].shift(4)-1)*100
    df_q['pctR5'] = (df_q["close"]/df_q["close"].shift(5)-1)*100
    df_q['pctR6'] = (df_q["close"]/df_q["close"].shift(6)-1)*100
    df_q['pctR7'] = (df_q["close"]/df_q["close"].shift(7)-1)*100
    df_q['pctR8'] = (df_q["close"]/df_q["close"].shift(8)-1)*100

    df_q['kline'] = df_q.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_q['cp5'] = (df_q["close"] - df_q["close"].rolling(5,min_periods=1).min()) \
                       / (df_q["close"].rolling(5).max() - df_q["close"].rolling(5).min())
    df_q['cp10'] = (df_q["close"] - df_q["close"].rolling(10,min_periods=1).min()) \
                        / (df_q["close"].rolling(10).max() - df_q["close"].rolling(10).min())

    df_q['vt5'] = (df_q["amount"] - df_q["amount"].rolling(5,min_periods=1).min()) \
                       / (df_q["amount"].rolling(5).max() - df_q["amount"].rolling(5).min())
    df_q['vt10'] = (df_q["amount"] - df_q["amount"].rolling(10,min_periods=1).min()) \
                        / (df_q["amount"].rolling(10).max() - df_q["amount"].rolling(10).min())
    df_q['vratio5'] = df_q["amount"] / df_q["amount"].rolling(5,min_periods=1).mean()
    df_q['vratio10'] = df_q["amount"] / df_q["amount"].rolling(10,min_periods=1).mean()

    df_q['aft1D'] = df_q["close"].shift(-1)/df_q["close"]*100-100
    df_q['aft2D'] = df_q["close"].shift(-2)/df_q["close"]*100-100
    df_q['aft3D'] = df_q["close"].shift(-3)/df_q["close"]*100-100

    df_q['volm5'] = df_q['volume'].rolling(5).mean()
    df_q['volabovem5'] = (df_q.volume > df_q.volm5).map({True: 1, False: 0})
    df_q['bias'] = computeBias(df_q[['close']], 5)
    df_q['biasup'] = df_q['bias'] < df_q['bias'].shift(1)

    df_q['ma5'] = df_q["close"].rolling(5).mean()
    df_q['abovema5'] = (df_q["close"]>df_q['ma5']).map({True: 1, False: 0})
    _,_, df_q['macd'] = ta.MACD(df_q.close.values, 5,10,9)
    df_q['macdpos1st'] = ((df_q.macd>0) & (df_q.macd.shift(1)<0)).map({True: 1, False: 0})
    df_q['macdnearpos'] = ((df_q.macd<0) & (df_q.macd.shift(1)<df_q.macd.shift(2)/2)).map({True: 1, False: 0})
    # df_q['kpos1st'] = (df_q.kline>0 and df_q.kline.shift(1)==0 and df_q.kline.shift(2)==0).map({True: 1, False: 0})
    df_q['kpos1st'] = (df_q.kline > df_q.kline.shift(1)).map({True: 1, False: 0})
    df_q.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    df_q = df_q[(df_q['preclose']>0) & (df_q['open']>0)]

    df_q['code'] = code
    df_q['sname'] = name
    df_q.reset_index(drop=True, inplace=True)

    if len(df_q)>5:
        return df_q
    else:
        return pd.DataFrame()

# 数据获取四级函数 - TdxBK
def prepTdxBKMdata(code, name, df_status):
    # global today, stocklist

    if len(df_status)<200:
        return pd.DataFrame()

    df_status.index = pd.to_datetime(df_status['date'])

    df_q = pd.DataFrame()

    # df_q['open'] = df_status.open.resample('Q').apply(custom_resampler)
    df_q['open'] = df_status.open.resample('M').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['high'] = df_status.high.resample('M').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['low'] = df_status.low.resample('M').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['close'] = df_status.close.resample('M').apply(lambda x: x[-1] if len(x)>0 else 0)

    qleftdays = (df_q.index[-1] - df_status.index[-1]).days
    # df_q.turn[-1] = df_q.turn[-1] / (90 - qleftdays) * 90
    # df_q['turn'] = df_status.turn.resample('M').sum()
    df_q['volume'] = df_status.volume.resample('M').sum()/100000000
    df_q['amount'] = df_status.amount.resample('M').sum()/100000000

    df_q.volume[-1] = df_q.volume[-1] / (31 - qleftdays) * 31
    df_q.amount[-1] = df_q.amount[-1] / (31 - qleftdays) * 31

    df_q['date'] = df_q.index.strftime('%Y-%m-%d')
    df_q['preclose'] = df_q['close'].shift(1)

    df_q['opct'] = (df_q['open'] / df_q['preclose'] - 1) * 100
    df_q['hpct'] = (df_q['high'] / df_q['preclose'] - 1) * 100
    df_q['lpct'] = (df_q['low'] / df_q['preclose'] - 1) * 100

    df_q['pctChg'] = (df_q["close"]/df_q["close"].shift(1)-1)*100
    df_q['pctR2'] = (df_q["close"]/df_q["close"].shift(2)-1)*100
    df_q['pctR3'] = (df_q["close"]/df_q["close"].shift(3)-1)*100
    df_q['pctR4'] = (df_q["close"]/df_q["close"].shift(4)-1)*100
    df_q['pctR5'] = (df_q["close"]/df_q["close"].shift(5)-1)*100
    df_q['pctR6'] = (df_q["close"]/df_q["close"].shift(6)-1)*100
    df_q['pctR7'] = (df_q["close"]/df_q["close"].shift(7)-1)*100
    df_q['pctR8'] = (df_q["close"]/df_q["close"].shift(8)-1)*100

    df_q['kline'] = df_q.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_q['cp5'] = (df_q["close"] - df_q["close"].rolling(5,min_periods=1).min()) \
                       / (df_q["close"].rolling(5).max() - df_q["close"].rolling(5).min())
    df_q['cp10'] = (df_q["close"] - df_q["close"].rolling(10,min_periods=1).min()) \
                        / (df_q["close"].rolling(10).max() - df_q["close"].rolling(10).min())
    df_q['cp20'] = (df_q["close"] - df_q["close"].rolling(20,min_periods=1).min()) \
                        / (df_q["close"].rolling(20).max() - df_q["close"].rolling(20).min())
    df_q['vt5'] = (df_q["amount"] - df_q["amount"].rolling(5,min_periods=1).min()) \
                       / (df_q["amount"].rolling(5).max() - df_q["amount"].rolling(5).min())
    df_q['vt10'] = (df_q["amount"] - df_q["amount"].rolling(10,min_periods=1).min()) \
                        / (df_q["amount"].rolling(10).max() - df_q["amount"].rolling(10).min())
    df_q['vt20'] = (df_q["amount"] - df_q["amount"].rolling(20,min_periods=1).min()) \
                        / (df_q["amount"].rolling(20).max() - df_q["amount"].rolling(20).min())
    df_q['vratio5'] = df_q["amount"] / df_q["amount"].rolling(5,min_periods=1).mean()
    df_q['vratio10'] = df_q["amount"] / df_q["amount"].rolling(10,min_periods=1).mean()

    df_q['aft1D'] = df_q["close"].shift(-1)/df_q["close"]*100-100
    df_q['aft2D'] = df_q["close"].shift(-2)/df_q["close"]*100-100
    df_q['aft3D'] = df_q["close"].shift(-3)/df_q["close"]*100-100

    df_q['volm5'] = df_q['volume'].rolling(5).mean()
    df_q['volabovem5'] = (df_q.volume > df_q.volm5).map({True: 1, False: 0})
    df_q['bias'] = computeBias(df_q[['close']], 5)
    df_q['biasup'] = df_q['bias'] < df_q['bias'].shift(1)

    df_q['ma5'] = df_q["close"].rolling(5).mean()
    df_q['abovema5'] = (df_q["close"]>df_q['ma5']).map({True: 1, False: 0})
    _,_, df_q['macd'] = ta.MACD(df_q.close.values, 5,10,9)
    df_q['macdpos1st'] = ((df_q.macd>0) & (df_q.macd.shift(1)<0)).map({True: 1, False: 0})
    df_q['macdnearpos'] = ((df_q.macd<0) & (df_q.macd.shift(1)<df_q.macd.shift(2)/2)).map({True: 1, False: 0})
    # df_q['kpos1st'] = (df_q.kline>0 and df_q.kline.shift(1)==0 and df_q.kline.shift(2)==0).map({True: 1, False: 0})
    df_q['kpos1st'] = (df_q.kline > df_q.kline.shift(1)).map({True: 1, False: 0})
    df_q.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    df_q = df_q[(df_q['preclose']>0) & (df_q['open']>0)]

    df_q['c20cls'] = df_q.apply(lambda x: mybin(x.cp20), axis=1)
    df_q['v20cls'] = df_q.apply(lambda x: mybin(x.vt20), axis=1)
    df_q['cv'] =  df_q['c20cls'] * 10 + df_q['v20cls']

    df_q['code'] = code
    df_q['sname'] = name
    df_q.reset_index(drop=True, inplace=True)

    if len(df_q)>5:
        return df_q
    else:
        return pd.DataFrame()

# 数据获取四级函数 - TdxBK
def prepTdxBKWdata(code, name, df_status):
    global today, stocklist

    if len(df_status)<200:
        return pd.DataFrame()

    df_status.index = pd.to_datetime(df_status['date'])

    df_q = pd.DataFrame()

    # df_q['open'] = df_status.open.resample('Q').apply(custom_resampler)
    df_q['open'] = df_status.open.resample('W').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['high'] = df_status.high.resample('W').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['low'] = df_status.low.resample('W').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['close'] = df_status.close.resample('W').apply(lambda x: x[-1] if len(x)>0 else 0)

    df_q = df_q[df_q['close'] > 0]
    qleftdays = df_status.index[-1].weekday()
    # df_q.turn[-1] = df_q.turn[-1] / (90 - qleftdays) * 90
    # df_q['turn'] = df_status.turn.resample('M').sum()
    df_q['volume'] = df_status.volume.resample('W').sum()/100000000
    df_q['amount'] = df_status.amount.resample('W').sum()/100000000

    df_q.volume[-1] = df_q.volume[-1] / (qleftdays+1) * 5
    df_q.amount[-1] = df_q.amount[-1] / (qleftdays+1) * 5

    df_q['date'] = df_q.index.strftime('%Y-%m-%d')
    df_q['preclose'] = df_q['close'].shift(1)

    df_q['opct'] = (df_q['open'] / df_q['preclose'] - 1) * 100
    df_q['hpct'] = (df_q['high'] / df_q['preclose'] - 1) * 100
    df_q['lpct'] = (df_q['low'] / df_q['preclose'] - 1) * 100

    df_q['pctChg'] = (df_q["close"]/df_q["close"].shift(1)-1)*100
    df_q['pctR2'] = (df_q["close"]/df_q["close"].shift(2)-1)*100
    df_q['pctR3'] = (df_q["close"]/df_q["close"].shift(3)-1)*100
    df_q['pctR4'] = (df_q["close"]/df_q["close"].shift(4)-1)*100
    df_q['pctR5'] = (df_q["close"]/df_q["close"].shift(5)-1)*100
    df_q['pctR6'] = (df_q["close"]/df_q["close"].shift(6)-1)*100
    df_q['pctR7'] = (df_q["close"]/df_q["close"].shift(7)-1)*100
    df_q['pctR8'] = (df_q["close"]/df_q["close"].shift(8)-1)*100

    df_q['kline'] = df_q.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_q['cp5'] = (df_q["close"] - df_q["close"].rolling(5,min_periods=1).min()) \
                       / (df_q["close"].rolling(5).max() - df_q["close"].rolling(5).min())
    df_q['cp10'] = (df_q["close"] - df_q["close"].rolling(10,min_periods=1).min()) \
                        / (df_q["close"].rolling(10).max() - df_q["close"].rolling(10).min())

    df_q['vt5'] = (df_q["amount"] - df_q["amount"].rolling(5,min_periods=1).min()) \
                       / (df_q["amount"].rolling(5).max() - df_q["amount"].rolling(5).min())
    df_q['vt10'] = (df_q["amount"] - df_q["amount"].rolling(10,min_periods=1).min()) \
                        / (df_q["amount"].rolling(10).max() - df_q["amount"].rolling(10).min())
    df_q['vratio5'] = df_q["amount"] / df_q["amount"].rolling(5,min_periods=1).mean()
    df_q['vratio10'] = df_q["amount"] / df_q["amount"].rolling(10,min_periods=1).mean()

    df_q['aft1D'] = df_q["close"].shift(-1)/df_q["close"]*100-100
    df_q['aft2D'] = df_q["close"].shift(-2)/df_q["close"]*100-100
    df_q['aft3D'] = df_q["close"].shift(-3)/df_q["close"]*100-100

    df_q['volm5'] = df_q['volume'].rolling(5).mean()
    df_q['volabovem5'] = (df_q.volume > df_q.volm5).map({True: 1, False: 0})
    df_q['bias'] = computeBias(df_q[['close']], 5)
    df_q['biasup'] = df_q['bias'] < df_q['bias'].shift(1)

    df_q['ma5'] = df_q["close"].rolling(5).mean()
    df_q['abovema5'] = (df_q["close"]>df_q['ma5']).map({True: 1, False: 0})
    _,_, df_q['macd'] = ta.MACD(df_q.close.values, 5,10,9)
    df_q['macdpos1st'] = ((df_q.macd>0) & (df_q.macd.shift(1)<0)).map({True: 1, False: 0})
    df_q['macdnearpos'] = ((df_q.macd<0) & (df_q.macd.shift(1)<df_q.macd.shift(2)/2)).map({True: 1, False: 0})
    # df_q['kpos1st'] = (df_q.kline>0 and df_q.kline.shift(1)==0 and df_q.kline.shift(2)==0).map({True: 1, False: 0})
    df_q['kpos1st'] = (df_q.kline > df_q.kline.shift(1)).map({True: 1, False: 0})
    df_q.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    df_q = df_q[(df_q['preclose']>0) & (df_q['open']>0)]

    df_q['code'] = code
    df_q['sname'] = name
    df_q.reset_index(drop=True, inplace=True)

    if len(df_q)>5:
        return df_q
    else:
        return pd.DataFrame()

# 数据获取四级函数 - TdxBK
def prepTdxBKDdata(code, name, df_status, rank_days=40):
    df_status['code'] = code
    df_status['sname'] = name

    if 'volume' not in df_status.columns:
        df_status['volume'] = df_status['vol']

    df_status['preclose'] = df_status['close'].shift(1)

    df_status['amount'] = df_status['amount']/100000000

    df_status['opct'] = (df_status['open'] / df_status['preclose'] - 1) * 100
    df_status['hpct'] = (df_status['high'] / df_status['preclose'] - 1) * 100
    df_status['lpct'] = (df_status['low'] / df_status['preclose'] - 1) * 100

    df_status['pctChg'] = (df_status["close"]/df_status["close"].shift(1)-1)*100
    df_status['pctR2'] = (df_status["close"]/df_status["close"].shift(2)-1)*100
    df_status['pctR3'] = (df_status["close"]/df_status["close"].shift(3)-1)*100
    df_status['pctR4'] = (df_status["close"]/df_status["close"].shift(4)-1)*100
    df_status['pctR5'] = (df_status["close"]/df_status["close"].shift(5)-1)*100
    df_status['pctR6'] = (df_status["close"]/df_status["close"].shift(6)-1)*100
    df_status['pctR7'] = (df_status["close"]/df_status["close"].shift(7)-1)*100
    df_status['pctR8'] = (df_status["close"]/df_status["close"].shift(8)-1)*100

    df_status['kline'] = df_status.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_status['cp5'] = (df_status["close"] - df_status["close"].rolling(5,min_periods=1).min()) \
                       / (df_status["close"].rolling(5).max() - df_status["close"].rolling(5).min())
    df_status['cp20'] = (df_status["close"] - df_status["close"].rolling(20,min_periods=1).min()) \
                        / (df_status["close"].rolling(20).max() - df_status["close"].rolling(20).min())
    df_status['cp60'] = (df_status["close"] - df_status["close"].rolling(60,min_periods=1).min()) \
                        / (df_status["close"].rolling(60).max() - df_status["close"].rolling(60).min())

    df_status['vt5'] = (df_status["volume"] - df_status["volume"].rolling(5,min_periods=1).min()) \
                       / (df_status["volume"].rolling(5).max() - df_status["volume"].rolling(5).min())
    df_status['vt10'] = (df_status["volume"] - df_status["volume"].rolling(10,min_periods=1).min()) \
                        / (df_status["volume"].rolling(10).max() - df_status["volume"].rolling(10).min())
    df_status['vt20'] = (df_status["volume"] - df_status["volume"].rolling(20,min_periods=1).min()) \
                        / (df_status["volume"].rolling(20).max() - df_status["volume"].rolling(20).min())
    df_status['vt60'] = (df_status["volume"] - df_status["volume"].rolling(60,min_periods=1).min()) \
                        / (df_status["volume"].rolling(60).max() - df_status["volume"].rolling(60).min())
    df_status['vratio5'] = df_status["volume"] / df_status["volume"].rolling(5,min_periods=1).mean()
    df_status['vratio10'] = df_status["volume"] / df_status["volume"].rolling(10,min_periods=1).mean()
    df_status['vratio20'] = df_status["volume"] / df_status["volume"].rolling(20,min_periods=1).mean()
    df_status['vratio60'] = df_status["volume"] / df_status["volume"].rolling(60,min_periods=1).mean()

    df_status['aft1D'] = df_status["close"].shift(-1)/df_status["close"]*100-100
    # df_status['aft2D'] = df_status["close"].shift(-2)/df_status["close"]*100-100
    df_status['aft3D'] = df_status["close"].shift(-3)/df_status["close"]*100-100
    # df_status['aft4D'] = df_status["pctChg"].rolling(4).sum().shift(-4)
    df_status['aft5D'] = df_status["close"].shift(-5)/df_status["close"]*100-100
    # df_status['aft8D'] = df_status["close"].shift(-8)/df_status["close"]*100-100
    # df_status['aft10D'] = df_status["close"].shift(-10)/df_status["close"]*100-100

    df_status['rank'] = df_status['volume'].rolling(rank_days + 1).apply(lambda x: pd.Series(x).rank().iloc[-1])
    df_status['vol_rank'] = 2 * (df_status['rank'] - rank_days - 1) / rank_days + 1
    df_status['bias'] = computeBias(df_status[['close']], 10)
    df_status['biasgood'] = (df_status['bias']<0).map({True: 1, False: 0})
    df_status['biasup'] = df_status['bias'] < df_status['bias'].shift(1)

    df_status['ma5'] = df_status["close"].rolling(5).mean()
    df_status['ma5up'] = (df_status['ma5']>df_status['ma5'].shift(1))
    df_status['ma5delta'] = (df_status['ma5']/df_status['ma5'].shift(1)-1)*100
    df_status.ma5up = df_status.ma5up.map({True: 1, False: 0})
    df_status['abovema5'] = (df_status["close"]>df_status['ma5']).map({True: 1, False: 0})

    df_status['ma10'] = df_status["close"].rolling(10).mean()
    df_status['ma10up'] = df_status['ma10']>df_status['ma10'].shift(1)
    df_status['ma10delta'] = (df_status['ma10'] / df_status['ma10'].shift(1) - 1) * 100
    df_status.ma10up = df_status.ma10up.map({True: 1, False: 0})
    df_status['abovema10'] = (df_status["close"] > df_status['ma10']).map({True: 1, False: 0})

    df_status['kline'] = df_status.apply(lambda x: 1 if x.ma10up and x.close>x.ma10 else 0, axis=1)

    df_status['ma20'] = df_status["close"].rolling(20).mean()
    df_status['ma20up'] = df_status['ma20']>df_status['ma20'].shift(1)
    df_status.ma20up = df_status.ma20up.map({True: 1, False: 0})
    df_status['ma20delta'] = (df_status['ma20'] / df_status['ma20'].shift(1) - 1) * 100
    df_status['abovema20'] = (df_status["close"] > df_status['ma20']).map({True: 1, False: 0})

    df_status['indicator'] = df_status.apply(lambda x: calcIndicator(x.ma5, x.ma10, x.ma20), axis=1)

    df_status['kdj'] = computeKDJv2(df_status.copy())
    df_status['bwidth'], df_status['bupperdelta'] = computeBollv2(df_status.copy())
    # df_status['rsi'] = computeRSI(df_status)
    df_status['rsi'] = ta.RSI(df_status['close'], timeperiod=14)
    # df_status['obv'] = computeOBV(df_status)
    df_status['obv'] = ta.OBV(df_status['close'].values, df_status['volume'].values)
    df_status['obv'] = (df_status['obv']>df_status['obv'].shift(2)).map({True: 1, False: 0})

    df_status['week'] = df_status.apply(lambda x: week_day(x.date), axis=1)
    df_status['c60cls'] = df_status.apply(lambda x: mybin(x.cp60), axis=1)
    df_status['v60cls'] = df_status.apply(lambda x: mybin(x.vt60), axis=1)
    df_status['cv'] = df_status['c60cls'] * 10 + df_status['v60cls']

    return df_status

# 数据获取一级函数 - stocks
def getStkdata(data_pull_s, data_pull_e, fn_tag, freq, qty=800):

    print('\n===== start 4k+ stock data pulling ======')
    print('data_pull_s:', data_pull_s, 'data_pull_e:', data_pull_e, 'Freq:', freq)

    sl = pd.read_csv('masterdata\\stocklist_full.csv', dtype=str, encoding='utf8')
    np.random.seed(13)
    rows = np.random.choice(sl.index.values, min(qty,len(sl)))
    sl = sl.loc[rows]
    print("\n股票数量：", len(sl))
    batchCnt = 8
    stockPerBatch = len(sl) // batchCnt + 1
    stockBatchList = []

    for i in range(batchCnt):
        sl_temp = sl[i * stockPerBatch: (i + 1) * stockPerBatch]
        stocklist_temp = {}
        for index, row in sl_temp.iterrows():
            stocklist_temp[row['code']] = row['sname']
        stockBatchList.append(stocklist_temp)

    df_concat = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor() as executor:  # 多进程
        results = [executor.submit(processStkBatches, i, batch, data_pull_s, data_pull_e, freq, fn_tag) \
                   for i, batch in enumerate(stockBatchList)]
        df_concat = pd.concat([result.result() for result in results], ignore_index=True)

    # df_concat = processStkBatches(0, stockBatchList[1], data_pull_s, data_pull_e, freq, fn_tag)
    # print(df_concat.columns)

    df_concat.loc[(df_concat["pctChg"] < 9.8) & (df_concat["pctChg"] > -9.8), 'zt'] = 'N'
    df_concat['zt'] = df_concat['zt'].fillna('Y')

    return df_concat

# 数据获取一级函数 - stocks
def getStkdataAllPeriods(ds, de, qty=800):

    print('\n===== start stock data pulling ======')

    sl = pd.read_csv('masterdata\\stocklist_full.csv', dtype=str, encoding='utf8')
    # np.random.seed(13)
    rows = np.random.choice(sl.index.values, min(qty,len(sl)))
    sl = sl.loc[rows]
    # print("\n股票数量：", len(sl))
    # sl = sl[sl['code']=='000026']

    batchCnt = 1
    stockPerBatch = len(sl) // batchCnt + 1
    stockBatchList = []

    for i in range(batchCnt):
        sl_temp = sl[i * stockPerBatch: (i + 1) * stockPerBatch]
        stocklist_temp = {}
        for index, row in sl_temp.iterrows():
            stocklist_temp[row['code']] = row['sname']
        stockBatchList.append(stocklist_temp)

    # df_concat = pd.DataFrame()
    # with concurrent.futures.ProcessPoolExecutor() as executor:  # 多进程
    #     # results = [executor.submit(processStkBatches, i, batch, data_pull_s, data_pull_e, freq, fn_tag) \
    #     results = [executor.submit(processStkBatchesAllPeriods, i, batch, min_start, ds, ws, ms, qs) \
    #                for i, batch in enumerate(stockBatchList)]
    #     # df_concatD = pd.concat([result.result() for result in results], ignore_index=True)
    #     df_concatD = pd.concat([result.result()[0] for result in results], ignore_index=True)
    #     df_concatW = pd.concat([result.result()[1] for result in results], ignore_index=True)
    #     df_concatM = pd.concat([result.result()[2] for result in results], ignore_index=True)
    #     df_concatQ = pd.concat([result.result()[3] for result in results], ignore_index=True)

    df_concat = processStkBatchesAllPeriods(0, stockBatchList[0], ds,de)

    return df_concat

def getStkdataQonly(ds, de, qty=800):

    print('\n===== start stock data pulling ======')

    sl = pd.read_csv('masterdata\\stocklist_full.csv', dtype=str, encoding='utf8')
    # np.random.seed(13)
    # rows = np.random.choice(sl.index.values, min(qty,len(sl)))
    # sl = sl.loc[rows]
    # print("\n股票数量：", len(sl))
    # sl = sl[sl['code']=='000026']

    batchCnt = 8
    stockPerBatch = len(sl) // batchCnt + 1
    stockBatchList = []

    for i in range(batchCnt):
        sl_temp = sl[i * stockPerBatch: (i + 1) * stockPerBatch]
        stocklist_temp = {}
        for index, row in sl_temp.iterrows():
            stocklist_temp[row['code']] = row['sname']
        stockBatchList.append(stocklist_temp)

    df_concat = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor() as executor:  # 多进程
        # results = [executor.submit(processStkBatches, i, batch, data_pull_s, data_pull_e, freq, fn_tag) \
        results = [executor.submit(processStkBatchesQonly, i, batch, ds, de) \
                   for i, batch in enumerate(stockBatchList)]
        df_concat = pd.concat([result.result() for result in results], ignore_index=True)

    # df_concat = processStkBatchesQonly(0, stockBatchList[0], ds,de)

    return df_concat

# 数据获取二级函数 - stocks
def processStkBatches(seq, batch, data_pull_s, freq, fn_tag):
    global conf

    api = TdxHq_API()
    if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport )==False:
        print('connection to TDX server not available')

    print("processing batch ", seq, "Frequency:", freq)
    # codelist = batch.keys()

    # if (fn_tag[:3].lower() == 'stk') or ('_stk_' in fn_tag):
    #     # print('get stk1d data with features for prog_tag', fn_tag)
    #     stkFeatures = True

    batch_start = time.time()
    df_batch = pd.DataFrame()

    counter = 0
    for key, value in batch.items():
        df_single = getStkDataFromBaoEM(key, value, data_pull_s, freq, api)
        if len(df_single) > 0:
            df_batch = pd.concat([df_batch, df_single])

        counter += 1
        if counter % 100 == 0:
            print('.' + str(counter))
        else:
            print('.', end='', flush=True)

    df_batch.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    if freq == 'D':
        print("batch ", seq, "before ZT removal, data rows:", len(df_batch))
        print("removing rows w/ pctChg outside of [-9.8, 9.8]")
        df_batch = df_batch[(df_batch['pctChg']>=-9.8) & (df_batch['pctChg']<=9.8)]
    print('\nbatch:', seq, 'rows:', len(df_batch), ' Data Prepared in ', round(time.time() - batch_start, 0), ' secs.')

    api.close()
    return df_batch

# 数据获取二级函数 - stocks
def processStkBatchesAllPeriods(seq, batch, ds, de):

    api = TdxHq_API()
    if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport )==False:
        print('connection to TDX server not available')

    batch_start = time.time()


    counter = 0
    df_batch = pd.DataFrame()
    for key, value in batch.items():
        df_single = getStkDataFromBaoEMAllPeriods(key, value, ds, api)
        df_batch = pd.concat([df_batch, df_single])
        counter += 1
        if counter % 100 == 0:
            print('.' + str(counter))
        else:
            print('.', end='', flush=True)

    # df_batchD.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    # print("batch ", seq, "before ZT removal, data rows:", len(df_batchD))
    # print("removing rows w/ pctChg outside of [-9.8, 9.8]")
    # df_batchD = df_batchD[(df_batchD['pctChg']>=-9.8) & (df_batchD['pctChg']<=9.8)]
    print('\nbatch:', seq, 'rows:', len(df_batch), ' Data Prepared in ', round(time.time() - batch_start, 0), ' secs.')

    api.close()

    return df_batch

# 数据获取二级函数 - stocks
def processStkBatchesQonly(seq, batch, ds, de):

    api = TdxHq_API()
    if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport )==False:
        print('connection to TDX server not available')

    batch_start = time.time()


    counter = 0
    df_batch = pd.DataFrame()
    for key, value in batch.items():
        df_single = getStkDataFromBaoEMQonly(key, value, ds, api)
        df_batch = pd.concat([df_batch, df_single])
        counter += 1
        if counter % 100 == 0:
            print('.' + str(counter))
        else:
            print('.', end='', flush=True)

    # df_batchD.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    # print("batch ", seq, "before ZT removal, data rows:", len(df_batchD))
    # print("removing rows w/ pctChg outside of [-9.8, 9.8]")
    # df_batchD = df_batchD[(df_batchD['pctChg']>=-9.8) & (df_batchD['pctChg']<=9.8)]
    print('\nbatch:', seq, 'rows:', len(df_batch), ' Data Prepared in ', round(time.time() - batch_start, 0), ' secs.')

    api.close()

    return df_batch

# 数据获取三级函数 - stocks
def getStkDataFromBaoEM(code, name, data_pull_s, freq, api):

    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}

    if code[:1].lower() == 's':
        print(code, 'skipped.')
        return pd.DataFrame()
    else:
        BaoDataFile = bao_data_path + 'BAO-' + code +'.csv'
        if os.path.exists(BaoDataFile):
            df_code = pd.read_csv(BaoDataFile, dtype=dtype_dic, encoding='gbk')
            # df_single = df_code[(df_code['date'] >= data_pull_s) & (df_code['date'] <= data_pull_e)]
            df_single = df_code[(df_code['date'] >= data_pull_s)]
            df_single['change'] = df_single['pctChg'] / 100
            df_single[['open', 'close', 'high', 'low']] = cal_right_price(df_single, type='后复权')

            if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
                    and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:

                # # get realtime data from EM web.
                # newdata = pullRealtimeDataFromEMweb(code)
                # if len(newdata) == 1 and len(df_single) > 20:
                #     print("+", end='', flush=False)
                #     # df_status = df_status.append(newdata[0], ignore_index=True)
                #     df_single = df_single.append({
                #         'date': newdata[0]['date'], 'code': newdata[0]['code'],
                #         'open': newdata[0]['open'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                #         'high': newdata[0]['high'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                #         'low': newdata[0]['low'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                #         'close': newdata[0]['close'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                #         'volume': newdata[0]['volume'], 'amount': newdata[0]['amount'],
                #         'turn': newdata[0]['turn'], 'pctChg': newdata[0]['pctChg'], 'change': newdata[0]['pctChg'] / 100
                #     }, ignore_index=True)
                # else:
                #     print(code,'rt failed')

                # get realtime data from Tdx.
                try:
                    if code[0] == '6':
                        market = 1
                    else:
                        market = 0
                    date, close, amount, vol, high, low, open, preclose = getTdxRealtime(api, market, code)
                    print("+", end='', flush=False)
                    # df_status = df_status.append(newdata[0], ignore_index=True)
                    df_single = df_single.append({
                        'date': date, 'code': code,
                        'open': open / preclose * list(df_single['close'])[-1],
                        'high': high / preclose * list(df_single['close'])[-1],
                        'low': low / preclose * list(df_single['close'])[-1],
                        'close': close / preclose * list(df_single['close'])[-1],
                        'volume': vol, 'amount': amount,
                        'turn': 0, 'pctChg': (close/preclose-1)*100, 'change': (close/preclose-1)
                    }, ignore_index=True)
                except:
                    print(code,'TdxRealtime failed.')

                try:
                    if list(df_single.date)[-1] == datetime.datetime.now().strftime('%Y-%m-%d'):
                        df_single.turn[len(df_single) - 1] = df_single.volume[len(df_single) - 1] / df_single.volume[
                            len(df_single) - 2] * df_single.turn[len(df_single) - 2]
                except:
                    pass
            else:
                pass    # non-dealing hours
        else:
            print(code,'no baofile.')
            return pd.DataFrame()

        if freq == 'Q':
            df_single = prepStkQdata(code, name, df_single)
            return df_single
        elif freq == "M":
            df_single = prepStkMdata(code, name, df_single)
            return df_single
        elif freq == 'W':
            df_single = prepStkWdata(code, name, df_single)
            return df_single
        elif freq == 'D':
            df_single = prepStkDdata(code, name, df_single)
            return df_single
        else:
            print('freq ', freq, 'error.  it should be Q/M/W/D')
            return pd.DataFrame()

def getStkDataFromBaoEMAllPeriods(code, name, data_pull_s, api):

    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}

    if code[:1].lower() == 's':
        print(code, 'skipped.')
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    else:
        BaoDataFile = bao_data_path + 'BAO-' + code +'.csv'
        if os.path.exists(BaoDataFile):
            df_code = pd.read_csv(BaoDataFile, dtype=dtype_dic, encoding='gbk')
            # df_single = df_code[(df_code['date'] >= data_pull_s) & (df_code['date'] <= data_pull_e)]
            df_single = df_code[(df_code['date'] >= data_pull_s)]
            df_single['change'] = df_single['pctChg'] / 100
            df_single[['open', 'close', 'high', 'low']] = cal_right_price(df_single, type='后复权')

            if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
                    and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:

                # get realtime data from EM web.
                newdata = pullRealtimeDataFromEMweb(code)
                if len(newdata) == 1 and len(df_single) > 20:
                    print("+", end='', flush=False)
                    # df_status = df_status.append(newdata[0], ignore_index=True)
                    df_single = df_single.append({
                        'date': newdata[0]['date'], 'code': newdata[0]['code'],
                        'open': newdata[0]['open'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'high': newdata[0]['high'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'low': newdata[0]['low'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'close': newdata[0]['close'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'volume': newdata[0]['volume']*100, 'amount': newdata[0]['amount'],
                        'turn': newdata[0]['turn'], 'pctChg': newdata[0]['pctChg'], 'change': newdata[0]['pctChg'] / 100
                    }, ignore_index=True)
                else:
                    print(code,'rt failed')

                try:
                    if list(df_single.date)[-1] == datetime.datetime.now().strftime('%Y-%m-%d'):
                        df_single.turn[len(df_single) - 1] = df_single.volume[len(df_single) - 1] / df_single.volume[
                            len(df_single) - 2] * df_single.turn[len(df_single) - 2]
                except:
                    pass
            else:
                pass    # non-dealing hours
        else:
            print(code,'no baofile.')
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),

        df_singleD = prepStkDdata(code, name, df_single)
        df_all = df_singleD[79:]
        df_W = pd.DataFrame()
        df_M = pd.DataFrame()
        df_Q = pd.DataFrame()
        for i in range(len(df_singleD)):
            if i <80:
                continue
            elif i<len(df_singleD)-2:
                df_W = pd.concat([df_W, prepStkWdata(df_single[i-30:i])])
                df_M = pd.concat([df_M, prepStkMdata(df_single[:i])])
                df_Q = pd.concat([df_Q, prepStkQdata(df_single[:i])])
            elif i > len(df_singleD)-2:
                df_W = pd.concat([df_W, prepStkWdata(df_single[i-30:])])
                df_M = pd.concat([df_M, prepStkMdata(df_single)])
                df_Q = pd.concat([df_Q, prepStkQdata(df_single)])

        df_all = pd.merge(df_all, df_W, on='date', how='left')
        df_all = pd.merge(df_all, df_M, on='date', how='left')
        df_all = pd.merge(df_all, df_Q, on='date', how='left')

        df_all['code'] = code
        df_all['name'] = name

        return df_all

def getStkDataFromBaoEMQonly(code, name, data_pull_s, api):

    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}

    if code[:1].lower() == 's':
        print(code, 'skipped.')
        return pd.DataFrame()
    else:
        BaoDataFile = bao_data_path + 'BAO-' + code +'.csv'
        if os.path.exists(BaoDataFile):
            df_code = pd.read_csv(BaoDataFile, dtype=dtype_dic, encoding='gbk')
            # df_single = df_code[(df_code['date'] >= data_pull_s) & (df_code['date'] <= data_pull_e)]
            df_single = df_code[(df_code['date'] >= data_pull_s)]
            df_single['change'] = df_single['pctChg'] / 100
            df_single[['open', 'close', 'high', 'low']] = cal_right_price(df_single, type='后复权')

            if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
                    and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:

                # get realtime data from EM web.
                newdata = pullRealtimeDataFromEMweb(code)
                if len(newdata) == 1 and len(df_single) > 20:
                    print("+", end='', flush=False)
                    # df_status = df_status.append(newdata[0], ignore_index=True)
                    df_single = df_single.append({
                        'date': newdata[0]['date'], 'code': newdata[0]['code'],
                        'open': newdata[0]['open'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'high': newdata[0]['high'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'low': newdata[0]['low'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'close': newdata[0]['close'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
                        'volume': newdata[0]['volume']*100, 'amount': newdata[0]['amount'],
                        'turn': newdata[0]['turn'], 'pctChg': newdata[0]['pctChg'], 'change': newdata[0]['pctChg'] / 100
                    }, ignore_index=True)
                else:
                    print(code,'rt failed')

                try:
                    if list(df_single.date)[-1] == datetime.datetime.now().strftime('%Y-%m-%d'):
                        df_single.turn[len(df_single) - 1] = df_single.volume[len(df_single) - 1] / df_single.volume[
                            len(df_single) - 2] * df_single.turn[len(df_single) - 2]
                except:
                    pass
            else:
                pass    # non-dealing hours
        else:
            print(code,'no baofile.')
            return pd.DataFrame()

        if len(df_single) < 200:
            return pd.DataFrame()

        df_Q = prepStkQdata(df_single)
        df_Q['code'] = code
        df_Q['name'] = name

        return df_Q


# for stock all period draw only  20220404
def getSingleStockAllPeriods(code, name, data_s,data_e):

    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}

    if code[:1].lower() == 's':
        print(code, 'skipped.')
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    else:
        BaoDataFile = bao_data_path + 'BAO-' + code +'.csv'
        if os.path.exists(BaoDataFile):
            df_code = pd.read_csv(BaoDataFile, dtype=dtype_dic, encoding='gbk')
            # df_single = df_code[(df_code['date'] >= data_pull_s) & (df_code['date'] <= data_pull_e)]
            df_single = df_code[(df_code['date']>=data_s) & (df_code['date']<=data_e)]
            df_single['change'] = df_single['pctChg'] / 100
            df_single[['open', 'close', 'high', 'low']] = cal_right_price(df_single, type='前复权')

            # if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
            #         and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:
            #
            #     # get realtime data from EM web.
            #     newdata = pullRealtimeDataFromEMweb(code)
            #     if len(newdata) == 1 and len(df_single) > 20:
            #         print("+", end='', flush=False)
            #         # df_status = df_status.append(newdata[0], ignore_index=True)
            #         df_single = df_single.append({
            #             'date': newdata[0]['date'], 'code': newdata[0]['code'],
            #             'open': newdata[0]['open'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
            #             'high': newdata[0]['high'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
            #             'low': newdata[0]['low'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
            #             'close': newdata[0]['close'] / newdata[0]['preclose'] * list(df_single['close'])[-1],
            #             'volume': newdata[0]['volume'], 'amount': newdata[0]['amount'],
            #             'turn': newdata[0]['turn'], 'pctChg': newdata[0]['pctChg'], 'change': newdata[0]['pctChg'] / 100
            #         }, ignore_index=True)
            #     else:
            #         print(code,'rt failed')
            #
            #     # # get realtime data from Tdx.
            #     # try:
            #     #     if code[0] == '6':
            #     #         market = 1
            #     #     else:
            #     #         market = 0
            #     #     date, close, amount, vol, high, low, open, preclose = getTdxRealtime(api, market, code)
            #     #     print("+", end='', flush=False)
            #     #     # df_status = df_status.append(newdata[0], ignore_index=True)
            #     #     df_single = df_single.append({
            #     #         'date': date, 'code': code,
            #     #         'open': open / preclose * list(df_single['close'])[-1],
            #     #         'high': high / preclose * list(df_single['close'])[-1],
            #     #         'low': low / preclose * list(df_single['close'])[-1],
            #     #         'close': close / preclose * list(df_single['close'])[-1],
            #     #         'volume': vol, 'amount': amount,
            #     #         'turn': 0, 'pctChg': (close/preclose-1)*100, 'change': (close/preclose-1)
            #     #     }, ignore_index=True)
            #     # except:
            #     #     print(code,'TdxRealtime failed.')
            #
            #     try:
            #         if list(df_single.date)[-1] == datetime.datetime.now().strftime('%Y-%m-%d'):
            #             df_single.turn[len(df_single) - 1] = df_single.volume[len(df_single) - 1] / df_single.volume[
            #                 len(df_single) - 2] * df_single.turn[len(df_single) - 2]
            #     except:
            #         pass
            # else:
            #     pass    # non-dealing hours
        else:
            print(code,'no baofile.')
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),

        # df_singleD = prepStkDdata(code, name, df_single[df_single['date']])
        # df_singleW = prepStkWdata(code, name, df_single[df_single['date']])
        # df_singleM = prepStkMdata(code, name, df_single[df_single['date']])
        # df_singleQ = prepStkQdata(code, name, df_single[df_single['date']])

        return df_single


# 数据获取四级函数 - stocks - sina obsoleted
def pullRealtimeDataFromSina(code):

    if code[0] == '6':
        code = "sh" + code
    elif code[0]=='0' or code[0]=='3':
        code = "sz" + code

    url = 'http://hq.sinajs.cn/list=' + code.replace('.', '')

    try:
        res = requests.get(url, timeout=2)
        if res.status_code == 200 and len(res.text)>40:
            open = float(res.text.split(',')[1])
            high = float(res.text.split(',')[4])
            low = float(res.text.split(',')[5])
            close = float(res.text.split(',')[3])
            preclose = float(res.text.split(',')[2])
            if code=='sh.000001':
                volume = float(res.text.split(',')[8])*100*min2day(res.text.split(',')[31])
            else:
                volume = float(res.text.split(',')[8])*min2day(res.text.split(',')[31])
            amount = float(res.text.split(',')[9])*min2day(res.text.split(',')[31])
            turn = 0
            pctChg = float((close-preclose)/preclose*100)
            date = res.text.split(',')[30]

            return [{'date': date, 'code': code, 'open': open, 'high': high, 'low': low, 'close': close,# 'preclose': preclose,
                     'volume': volume, 'amount': amount, 'turn': turn, 'preclose':preclose,
                     'pctChg': pctChg}]
        else:
            print('sina_error',res.text)
            return []

    except requests.exceptions.RequestException:
        print('sina_error',code)
        return []

# 数据获取四级函数 - stocks
def pullRealtimeDataFromEMweb(code):

    if code[0] == '6':
        code = "1." + code
    elif code[0]=='0' or code[0]=='3':
        code = "0." + code
    elif code[:2].lower()=='sh':
        code = "1." + code[2:].replace('.','')
    elif code[:2].lower()=='sz':
        code = "0." + code[2:].replace('.','')
    elif code[:5].lower()=='sz399':
        code = "0." + code[2:].replace('.','')

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
            if isinstance(datajson['f46'],str) or isinstance(datajson['f47'],str):
                return []   # '-' look like wrong data, bypass
            open = datajson['f46']
            high = datajson['f44']
            low = datajson['f45']
            close = datajson['f43']
            preclose = datajson['f60']

            volume = datajson['f47']*min2day(datetime.datetime.now().strftime('%H:%M:%S'))
            amount = datajson['f48']*min2day(datetime.datetime.now().strftime('%H:%M:%S'))
            turn = 0
            pctChg = datajson['f170']
            date = datetime.datetime.now().strftime('%Y-%m-%d')

            return [{'date': date, 'code': code, 'open': open, 'high': high, 'low': low, 'close': close,# 'preclose': preclose,
                     'volume': volume, 'amount': amount, 'turn': turn, 'preclose':preclose,
                     'pctChg': pctChg}]
        else:
            print('sEM_error',res.text)
            return []

    except requests.exceptions.RequestException:
        print('sina_error',code)
        return []

# 数据获取四级函数 - stocks
def prepStkQdata(df_status):

    df_status.index = pd.to_datetime(df_status['date'])

    df_q = pd.DataFrame()
    df_q['qo'] = df_status.open.resample('Q').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['qh'] = df_status.high.resample('Q').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['ql'] = df_status.low.resample('Q').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['qc'] = df_status.close.resample('Q').apply(lambda x: x[-1] if len(x)>0 else 0)

    df_q['qpcthigh'] = (df_q['qh']/df_q['qc'].shift(1)-1)*100
    df_q['qpctlow'] = (df_q['ql']/df_q['qc'].shift(1)-1)*100
    df_q['qpct'] = (df_q['qc']/df_q['qc'].shift(1)-1)*100
    df_q['qpctr1'] = df_q['qpct'].shift(1)
    df_q['qpctaft1'] = df_q['qpct'].shift(-1)
    df_q['qpcthighaft1'] = df_q['qpcthigh'].shift(-1)

    df_q['qk'] = df_q.apply(lambda x: 1 if x.qc>x.qo else 0, axis=1)
    df_q['qkr1'] = df_q['qk'].shift(1)
    df_q['qkr2'] = df_q['qk'].shift(2)
    df_q['qksame'] = df_q.apply(lambda x: 1 if x.qk==x.qkr1 else 0, axis=1)

    df_q['qma5'] = df_q["qc"].rolling(5).mean()
    df_q['qabvma5'] = (df_q["qc"]>df_q['qma5']).map({True: 1, False: 0})
    df_q['qma5up'] = (df_q['qma5']>df_q['qma5'].shift(1)).map({True: 1, False: 0})

    df_q.reset_index(drop=False, inplace=True)

    return df_q

# 数据获取四级函数 - stocks
def prepStkMdata(df_status):

    lastdate = df_status['date'].values[-1]
    df_status.index = pd.to_datetime(df_status['date'])
    df_q = pd.DataFrame()

    df_q['mo'] = df_status.open.resample('M').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['mh'] = df_status.high.resample('M').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['ml'] = df_status.low.resample('M').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['mc'] = df_status.close.resample('M').apply(lambda x: x[-1] if len(x)>0 else 0)
    df_q['mpct'] = (df_q['mc']/df_q['mc'].shift(1)-1)*100
    df_q['mpct-1'] = (df_q['mc'].shift(1)/df_q['mc'].shift(2)-1)*100

    df_q['mk'] = df_q.apply(lambda x: 1 if x.mc>x.mo else 0, axis=1)
    df_q['mkr1'] = df_q['mk'].shift(1)

    df_q['mma5'] = df_q["mc"].rolling(5).mean()
    df_q['mabvma5'] = (df_q["mc"]>df_q['mma5']).map({True: 1, False: 0})

    df_q.reset_index(drop=True, inplace=True)
    df_q = df_q[-1:]
    df_q['date'] = lastdate

    return df_q

# 数据获取四级函数 - stocks
def prepStkWdata(df_status):

    lastdate = df_status['date'].values[-1]
    df_status.index = pd.to_datetime(df_status['date'])
    df_q = pd.DataFrame()

    df_q['wo'] = df_status.open.resample('W').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['wh'] = df_status.high.resample('W').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['wl'] = df_status.low.resample('W').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['wc'] = df_status.close.resample('W').apply(lambda x: x[-1] if len(x)>0 else 0)
    df_q['wpct'] = (df_q['wc']/df_q['wc'].shift(1)-1)*100
    df_q['wpct-1'] = (df_q['wc'].shift(1)/df_q['wc'].shift(2)-1)*100

    df_q = df_q[df_q['wc']>0]
    df_q['date'] = df_q.index.strftime('%Y-%m-%d')
    df_q['wk'] = df_q.apply(lambda x: 1 if x.wc>x.wo else 0, axis=1)
    df_q['wkr1'] = df_q['wk'].shift(1)
    df_q['wma5'] = df_q["wc"].rolling(5).mean()
    df_q['wabvma5'] = (df_q["wc"]>df_q['wma5']).map({True: 1, False: 0})

    df_q.reset_index(drop=True, inplace=True)
    df_q = df_q[-1:]
    df_q['date'] = lastdate

    return df_q

# 数据获取四级函数 - stocks
def prepStkDdata(code, name, df_status):

    if len(df_status)<100:
        return pd.DataFrame()

    df_status['weekday'] = pd.to_datetime(df_status.date)
    df_status['weekday'] = df_status['weekday'].apply(lambda x: x.weekday())
    df_status['aft1D'] = df_status["close"].shift(-1) / df_status["close"] * 100 - 100
    df_status['aft2D'] = df_status["close"].shift(-2) / df_status["close"] * 100 - 100
    df_status['aft3D'] = df_status["close"].shift(-3) / df_status["close"] * 100 - 100

    df_status['preclose'] = df_status['close'].shift(1)
    # df_status['amount'] = df_status['amount']/100000000
    df_status['opct'] = (df_status['open'] / df_status['preclose'] - 1) * 100
    df_status['hpct'] = (df_status['high'] / df_status['preclose'] - 1) * 100
    df_status['lpct'] = (df_status['low'] / df_status['preclose'] - 1) * 100
    df_status['dk'] = df_status.apply(lambda x: 1 if x.close>x.open else 0, axis=1)
    df_status['dk-1'] = df_status['dk'].shift(1)
    df_status['ma5'] = df_status["close"].rolling(5).mean()
    df_status['abovema5'] = (df_status["close"]>df_status['ma5']).map({True: 1, False: 0})

    return df_status

# 数据获取一级函数 - EMBK
def getEMBKdata(EMBKlist, data_pull_s, data_pull_e, batchcnt, freq):
    batchCnt = batchcnt
    stockPerBatch = len(EMBKlist) // batchCnt + 1
    stockBatchList = []
    # bk_black = ['次新股','含可转债','央企改革']
    EMBKlist.columns = ['bkcode', 'marketcode', 'bkname', 'flow']
    bk_black = ['MSCI中国','标准普尔','创业板综','富时罗素','沪股通','华为概念','机构重仓','融资融券','深成500','深股通','中证500',
                '昨日涨停_含一字','昨日涨停','昨日连板_含一字','昨日连板','昨日触板']

    for i in range(batchCnt):
        sl_temp = EMBKlist[i * stockPerBatch: (i + 1) * stockPerBatch]
        stocklist_temp = {}
        for index, row in sl_temp.iterrows():
            if row['bkname'] not in bk_black:
                stocklist_temp[row['bkcode']] = row['bkname']
            else:
                print(row['bkname'], 'skipped')

        stockBatchList.append(stocklist_temp)

    with concurrent.futures.ProcessPoolExecutor() as executor:  # 多进程
        results = [executor.submit(processEMBKBatches, i, batch, data_pull_s, data_pull_e, freq) \
                   for i, batch in enumerate(stockBatchList)]
        df_concat = pd.concat([result.result() for result in results], ignore_index=True)

    # df_concat = processEMBKBatches(0, stockBatchList[0], data_pull_s, data_pull_e, freq)

    return df_concat

# 数据获取二级函数 - EMBK
def processEMBKBatches(seq, batch, data_pull_s, data_pull_e, freq):

    print("processing batch ", seq, "Frequency:", freq)

    batch_start = time.time()
    df_batch = pd.DataFrame()

    counter = 0
    for key, value in batch.items():
        df_single = getEMBKDataFromEMweb(key, value, data_pull_s, freq)
        if len(df_single)>0:
            df_batch = pd.concat([df_batch, df_single])
        else:
            print(key, value, 'data length 0')

        counter += 1
        if counter % 100 == 0:
            print('.' + str(counter))
        else:
            print('.', end='', flush=True)

    df_batch.fillna({'aft1D': 0, 'aft3D': 0, 'aft5D': 0}, inplace=True)
    print('\nbatch:', seq, 'rows:', len(df_batch), ' Data Prepared in ', round(time.time() - batch_start, 0), ' secs.')

    return df_batch

# 数据获取三级函数 - EMBK
def getEMBKDataFromEMweb(code, name, datefrom, freq):

    EM_fn = embk_data_path + 'EMBK_' + code + '.csv'
    if os.path.exists(EM_fn):
        df_fn = pd.read_csv(EM_fn, encoding='gbk')
        if len(df_fn)<100:
            return pd.DataFrame()

        if datetime.datetime.now().weekday() < 5 and time.strftime("%H%M", time.localtime()) >= pullrealtime_start \
                and time.strftime("%H%M", time.localtime()) <= pullrealtime_end:
            try:
                open, close, high, low, vol, amount = getEMBKRealtimeFromEMweb(code)
                df_fn = df_fn[df_fn['date'] < datetime.datetime.now().strftime('%Y-%m-%d')]
                df_fn = df_fn.append(
                    {'date': datetime.datetime.now().strftime('%Y-%m-%d'), 'open': open, 'close': close,
                     'high': high, 'low': low, 'vol': vol, 'amount': amount}, ignore_index=True)
                print("+", end='', flush=False)
                if list(df_fn.date)[-1] == datetime.datetime.now().strftime('%Y-%m-%d'):
                    df_fn.turn[len(df_fn)-1] = df_fn.amount[len(df_fn)-1] / df_fn.amount[len(df_fn)-2] * df_fn.turn[len(df_fn) - 2]
                    df_fn.pctChg[len(df_fn)-1] = (df_fn.close[len(df_fn)-1] / df_fn.close[len(df_fn)-2]-1)*100
            except:
                pass
        else:
            pass    #  non-dealing hours
    else:
        print(code, name, EM_fn, 'file does not exists')
        return pd.DataFrame()

    if freq == 'Q':
        df_single = prepEMBKQdata(code, name, df_fn)
        return df_single
    elif freq == "M":
        df_single = prepEMBKMdata(code, name, df_fn)
        return df_single
    elif freq == 'W':
        df_single = prepEMBKWdata(code, name, df_fn)
        return df_single
    elif freq == 'D':
        df_single = prepEMBKDdata(code, name, df_fn)
        return df_single
    else:
        print('freq ', freq, 'error.  it should be Q/M/W/D')
        return pd.DataFrame()

# 数据获取四级函数 - EMBK
def getEMBKRealtimeFromEMweb(bkcode):
    url = 'http://push2.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery112406142175621622367_1615545163158&secid=90.' + bkcode + \
          '&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6%2Cf7%2Cf8%2Cf9%2Cf10%2Cf11%2Cf12%2Cf13&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&iscr=0&ndays=1&_=1615545163159'
    res = requests.get(url)
    try:
        data1 = json.loads(res.text[42:-2])['data']['trends']
    except:
        return pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1], columns=['date', 'open', 'close', 'high','low', 'vol','amount', 'preclose'])
    # min.drop(labels=['ref1', 'ref2','ref3', 'ref4','ref5', 'ref6'],axis=1,inplace=True)
    # min['time'] = min['time'].astype('datetime64[ns]')

    min['open'] = min['open'].astype(float)
    min['close'] = min['close'].astype(float)
    min['high'] = min['high'].astype(float)
    min['low'] = min['low'].astype(float)
    min['vol'] = min['vol'].astype(float)
    min['amount'] = min['amount'].astype(float)
    min['preclose'] = min['preclose'].astype(float)
    timestr = datetime.datetime.now().strftime('%H:%M:%S')
    vol = min['vol'].sum()* min2day(timestr)
    amount = min['amount'].sum()* min2day(timestr)
    return list(min['open'])[0], list(min['close'])[-1], min['high'].max(), min['low'].min(), vol, amount

# 数据获取四级函数 - EMBK
def prepEMBKQdata(code, name, df_status):
    # global today, stocklist

    if len(df_status)<200:
        return pd.DataFrame()

    df_status.index = pd.to_datetime(df_status['date'])

    df_q = pd.DataFrame()

    # df_q['open'] = df_status.open.resample('Q').apply(custom_resampler)
    df_q['open'] = df_status.open.resample('Q').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['high'] = df_status.high.resample('Q').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['low'] = df_status.low.resample('Q').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['close'] = df_status.close.resample('Q').apply(lambda x: x[-1] if len(x)>0 else 0)

    qleftdays = (df_q.index[-1] - df_status.index[-1]).days

    if 'volume' not in df_status.columns:
        df_status['volume'] = df_status['vol']
    df_q['volume'] = df_status.volume.resample('Q').sum()/100000000
    df_q['amount'] = df_status.amount.resample('Q').sum()/100000000

    df_q.volume[-1] = df_q.volume[-1] / max((90 - qleftdays),1) * 90
    df_q.amount[-1] = df_q.amount[-1] / max((90 - qleftdays),1) * 90

    df_q['date'] = df_q.index.strftime('%Y-%m-%d')
    df_q['preclose'] = df_q['close'].shift(1)

    df_q['opct'] = (df_q['open'] / df_q['preclose'] - 1) * 100
    df_q['hpct'] = (df_q['high'] / df_q['preclose'] - 1) * 100
    df_q['lpct'] = (df_q['low'] / df_q['preclose'] - 1) * 100

    df_q['pctChg'] = (df_q["close"]/df_q["close"].shift(1)-1)*100
    df_q['pctR2'] = (df_q["close"]/df_q["close"].shift(2)-1)*100
    df_q['pctR3'] = (df_q["close"]/df_q["close"].shift(3)-1)*100
    df_q['pctR4'] = (df_q["close"]/df_q["close"].shift(4)-1)*100
    df_q['pctR5'] = (df_q["close"]/df_q["close"].shift(5)-1)*100
    df_q['pctR6'] = (df_q["close"]/df_q["close"].shift(6)-1)*100
    df_q['pctR7'] = (df_q["close"]/df_q["close"].shift(7)-1)*100
    df_q['pctR8'] = (df_q["close"]/df_q["close"].shift(8)-1)*100

    df_q['kline'] = df_q.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_q['cp5'] = (df_q["close"] - df_q["close"].rolling(5,min_periods=1).min()) \
                       / (df_q["close"].rolling(5).max() - df_q["close"].rolling(5).min())
    df_q['cp10'] = (df_q["close"] - df_q["close"].rolling(10,min_periods=1).min()) \
                        / (df_q["close"].rolling(10).max() - df_q["close"].rolling(10).min())

    df_q['vt5'] = (df_q["amount"] - df_q["amount"].rolling(5,min_periods=1).min()) \
                       / (df_q["amount"].rolling(5).max() - df_q["amount"].rolling(5).min())
    df_q['vt10'] = (df_q["amount"] - df_q["amount"].rolling(10,min_periods=1).min()) \
                        / (df_q["amount"].rolling(10).max() - df_q["amount"].rolling(10).min())
    df_q['vratio5'] = df_q["amount"] / df_q["amount"].rolling(5,min_periods=1).mean()
    df_q['vratio10'] = df_q["amount"] / df_q["amount"].rolling(10,min_periods=1).mean()

    df_q['aft1D'] = df_q["close"].shift(-1)/df_q["close"]*100-100
    df_q['aft2D'] = df_q["close"].shift(-2)/df_q["close"]*100-100
    df_q['aft3D'] = df_q["close"].shift(-3)/df_q["close"]*100-100

    df_q['volm5'] = df_q['volume'].rolling(5).mean()
    df_q['volabovem5'] = (df_q.volume > df_q.volm5).map({True: 1, False: 0})
    df_q['bias'] = computeBias(df_q[['close']], 5)
    df_q['biasup'] = df_q['bias'] < df_q['bias'].shift(1)

    df_q['ma5'] = df_q["close"].rolling(5).mean()
    df_q['abovema5'] = (df_q["close"]>df_q['ma5']).map({True: 1, False: 0})
    _,_, df_q['macd'] = ta.MACD(df_q.close.values, 5,10,9)
    df_q['macdpos1st'] = ((df_q.macd>0) & (df_q.macd.shift(1)<0)).map({True: 1, False: 0})
    df_q['macdnearpos'] = ((df_q.macd<0) & (df_q.macd.shift(1)<df_q.macd.shift(2)/2)).map({True: 1, False: 0})
    # df_q['kpos1st'] = (df_q.kline>0 and df_q.kline.shift(1)==0 and df_q.kline.shift(2)==0).map({True: 1, False: 0})
    df_q['kpos1st'] = (df_q.kline > df_q.kline.shift(1)).map({True: 1, False: 0})
    df_q.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    df_q = df_q[(df_q['preclose']>0) & (df_q['open']>0)]

    df_q['code'] = code
    df_q['sname'] = name
    df_q['macd'].fillna(0, inplace=True)
    df_q.reset_index(drop=True, inplace=True)

    if len(df_q)>5:
        return df_q
    else:
        return pd.DataFrame()

# 数据获取四级函数 - EMBK
def prepEMBKMdata(code, name, df_status):
    global today, stocklist

    if len(df_status)<200:
        return pd.DataFrame()

    df_status.index = pd.to_datetime(df_status['date'])

    df_q = pd.DataFrame()

    # df_q['open'] = df_status.open.resample('Q').apply(custom_resampler)
    df_q['open'] = df_status.open.resample('M').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['high'] = df_status.high.resample('M').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['low'] = df_status.low.resample('M').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['close'] = df_status.close.resample('M').apply(lambda x: x[-1] if len(x)>0 else 0)

    qleftdays = (df_q.index[-1] - df_status.index[-1]).days
    # df_q.turn[-1] = df_q.turn[-1] / (90 - qleftdays) * 90
    # df_q['turn'] = df_status.turn.resample('M').sum()
    if 'volume' not in df_status.columns:
        df_status['volume'] = df_status['vol']
    df_q['volume'] = df_status.volume.resample('M').sum()/100000000
    df_q['amount'] = df_status.amount.resample('M').sum()/100000000

    df_q.volume[-1] = df_q.volume[-1] / (31 - qleftdays) * 31
    df_q.amount[-1] = df_q.amount[-1] / (31 - qleftdays) * 31

    df_q['date'] = df_q.index.strftime('%Y-%m-%d')
    df_q['preclose'] = df_q['close'].shift(1)

    df_q['opct'] = (df_q['open'] / df_q['preclose'] - 1) * 100
    df_q['hpct'] = (df_q['high'] / df_q['preclose'] - 1) * 100
    df_q['lpct'] = (df_q['low'] / df_q['preclose'] - 1) * 100

    df_q['pctChg'] = (df_q["close"]/df_q["close"].shift(1)-1)*100
    df_q['pctR2'] = (df_q["close"]/df_q["close"].shift(2)-1)*100
    df_q['pctR3'] = (df_q["close"]/df_q["close"].shift(3)-1)*100
    df_q['pctR4'] = (df_q["close"]/df_q["close"].shift(4)-1)*100
    df_q['pctR5'] = (df_q["close"]/df_q["close"].shift(5)-1)*100
    df_q['pctR6'] = (df_q["close"]/df_q["close"].shift(6)-1)*100
    df_q['pctR7'] = (df_q["close"]/df_q["close"].shift(7)-1)*100
    df_q['pctR8'] = (df_q["close"]/df_q["close"].shift(8)-1)*100

    df_q['kline'] = df_q.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_q['cp5'] = (df_q["close"] - df_q["close"].rolling(5,min_periods=1).min()) \
                       / (df_q["close"].rolling(5).max() - df_q["close"].rolling(5).min())
    df_q['cp10'] = (df_q["close"] - df_q["close"].rolling(10,min_periods=1).min()) \
                        / (df_q["close"].rolling(10).max() - df_q["close"].rolling(10).min())
    df_q['cp20'] = (df_q["close"] - df_q["close"].rolling(20,min_periods=1).min()) \
                        / (df_q["close"].rolling(20).max() - df_q["close"].rolling(20).min())
    df_q['vt5'] = (df_q["amount"] - df_q["amount"].rolling(5,min_periods=1).min()) \
                       / (df_q["amount"].rolling(5).max() - df_q["amount"].rolling(5).min())
    df_q['vt10'] = (df_q["amount"] - df_q["amount"].rolling(10,min_periods=1).min()) \
                        / (df_q["amount"].rolling(10).max() - df_q["amount"].rolling(10).min())
    df_q['vt20'] = (df_q["amount"] - df_q["amount"].rolling(20,min_periods=1).min()) \
                        / (df_q["amount"].rolling(20).max() - df_q["amount"].rolling(20).min())
    df_q['vratio5'] = df_q["amount"] / df_q["amount"].rolling(5,min_periods=1).mean()
    df_q['vratio10'] = df_q["amount"] / df_q["amount"].rolling(10,min_periods=1).mean()

    df_q['aft1D'] = df_q["close"].shift(-1)/df_q["close"]*100-100
    df_q['aft2D'] = df_q["close"].shift(-2)/df_q["close"]*100-100
    df_q['aft3D'] = df_q["close"].shift(-3)/df_q["close"]*100-100

    df_q['volm5'] = df_q['volume'].rolling(5).mean()
    df_q['volabovem5'] = (df_q.volume > df_q.volm5).map({True: 1, False: 0})
    df_q['bias'] = computeBias(df_q[['close']], 5)
    df_q['biasup'] = df_q['bias'] < df_q['bias'].shift(1)

    df_q['ma5'] = df_q["close"].rolling(5).mean()
    df_q['abovema5'] = (df_q["close"]>df_q['ma5']).map({True: 1, False: 0})
    _,_, df_q['macd'] = ta.MACD(df_q.close.values, 12,26,9)
    df_q['macdpos1st'] = ((df_q.macd>0) & (df_q.macd.shift(1)<0)).map({True: 1, False: 0})
    df_q['macdnearpos'] = ((df_q.macd<0) & (df_q.macd.shift(1)<df_q.macd.shift(2)/2)).map({True: 1, False: 0})
    # df_q['kpos1st'] = (df_q.kline>0 and df_q.kline.shift(1)==0 and df_q.kline.shift(2)==0).map({True: 1, False: 0})
    df_q['kpos1st'] = (df_q.kline > df_q.kline.shift(1)).map({True: 1, False: 0})
    df_q.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    df_q = df_q[(df_q['preclose']>0) & (df_q['open']>0)]
    df_q['c20cls'] = df_q.apply(lambda x: mybin(x.cp20), axis=1)
    df_q['v20cls'] = df_q.apply(lambda x: mybin(x.vt20), axis=1)
    df_q['cv'] =  df_q['c20cls'] * 10 + df_q['v20cls']

    df_q['code'] = code
    df_q['sname'] = name
    df_q.reset_index(drop=True, inplace=True)

    if len(df_q)>5:
        return df_q
    else:
        return pd.DataFrame()

# 数据获取四级函数 - EMBK
def prepEMBKWdata(code, name, df_status):
    global today, stocklist

    if len(df_status)<200:
        return pd.DataFrame()

    df_status.index = pd.to_datetime(df_status['date'])

    df_q = pd.DataFrame()

    # df_q['open'] = df_status.open.resample('Q').apply(custom_resampler)
    df_q['open'] = df_status.open.resample('W').apply(lambda x: x[0] if len(x)>0 else 0)
    df_q['high'] = df_status.high.resample('W').apply(lambda x: x.max() if len(x)>0 else 0)
    df_q['low'] = df_status.low.resample('W').apply(lambda x: x.min() if len(x)>0 else 0)
    df_q['close'] = df_status.close.resample('W').apply(lambda x: x[-1] if len(x)>0 else 0)

    df_q = df_q[df_q['close'] > 0]
    qleftdays = df_status.index[-1].weekday()
    # df_q.turn[-1] = df_q.turn[-1] / (90 - qleftdays) * 90
    # df_q['turn'] = df_status.turn.resample('M').sum()

    if 'volume' not in df_status.columns:
        df_status['volume'] = df_status['vol']
    df_q['volume'] = df_status.volume.resample('W').sum()/100000000
    df_q['amount'] = df_status.amount.resample('W').sum()/100000000

    df_q.volume[-1] = df_q.volume[-1] / (qleftdays+1) * 5
    df_q.amount[-1] = df_q.amount[-1] / (qleftdays+1) * 5

    df_q['date'] = df_q.index.strftime('%Y-%m-%d')
    df_q['preclose'] = df_q['close'].shift(1)

    df_q['opct'] = (df_q['open'] / df_q['preclose'] - 1) * 100
    df_q['hpct'] = (df_q['high'] / df_q['preclose'] - 1) * 100
    df_q['lpct'] = (df_q['low'] / df_q['preclose'] - 1) * 100

    df_q['pctChg'] = (df_q["close"]/df_q["close"].shift(1)-1)*100
    df_q['pctR2'] = (df_q["close"]/df_q["close"].shift(2)-1)*100
    df_q['pctR3'] = (df_q["close"]/df_q["close"].shift(3)-1)*100
    df_q['pctR4'] = (df_q["close"]/df_q["close"].shift(4)-1)*100
    df_q['pctR5'] = (df_q["close"]/df_q["close"].shift(5)-1)*100
    df_q['pctR6'] = (df_q["close"]/df_q["close"].shift(6)-1)*100
    df_q['pctR7'] = (df_q["close"]/df_q["close"].shift(7)-1)*100
    df_q['pctR8'] = (df_q["close"]/df_q["close"].shift(8)-1)*100

    df_q['kline'] = df_q.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_q['cp5'] = (df_q["close"] - df_q["close"].rolling(5,min_periods=1).min()) \
                       / (df_q["close"].rolling(5).max() - df_q["close"].rolling(5).min())
    df_q['cp10'] = (df_q["close"] - df_q["close"].rolling(10,min_periods=1).min()) \
                        / (df_q["close"].rolling(10).max() - df_q["close"].rolling(10).min())

    df_q['vt5'] = (df_q["amount"] - df_q["amount"].rolling(5,min_periods=1).min()) \
                       / (df_q["amount"].rolling(5).max() - df_q["amount"].rolling(5).min())
    df_q['vt10'] = (df_q["amount"] - df_q["amount"].rolling(10,min_periods=1).min()) \
                        / (df_q["amount"].rolling(10).max() - df_q["amount"].rolling(10).min())
    df_q['vratio5'] = df_q["amount"] / df_q["amount"].rolling(5,min_periods=1).mean()
    df_q['vratio10'] = df_q["amount"] / df_q["amount"].rolling(10,min_periods=1).mean()

    df_q['aft1D'] = df_q["close"].shift(-1)/df_q["close"]*100-100
    df_q['aft2D'] = df_q["close"].shift(-2)/df_q["close"]*100-100
    df_q['aft3D'] = df_q["close"].shift(-3)/df_q["close"]*100-100

    df_q['volm5'] = df_q['volume'].rolling(5).mean()
    df_q['volabovem5'] = (df_q.volume > df_q.volm5).map({True: 1, False: 0})
    df_q['bias'] = computeBias(df_q[['close']], 5)
    df_q['biasup'] = df_q['bias'] < df_q['bias'].shift(1)

    df_q['ma5'] = df_q["close"].rolling(5).mean()
    df_q['abovema5'] = (df_q["close"]>df_q['ma5']).map({True: 1, False: 0})
    _,_, df_q['macd'] = ta.MACD(df_q.close.values, 12,26,9)
    df_q['macdpos1st'] = ((df_q.macd>0) & (df_q.macd.shift(1)<0)).map({True: 1, False: 0})
    df_q['macdnearpos'] = ((df_q.macd<0) & (df_q.macd.shift(1)<df_q.macd.shift(2)/2)).map({True: 1, False: 0})
    # df_q['kpos1st'] = (df_q.kline>0 and df_q.kline.shift(1)==0 and df_q.kline.shift(2)==0).map({True: 1, False: 0})
    df_q['kpos1st'] = (df_q.kline > df_q.kline.shift(1)).map({True: 1, False: 0})
    df_q.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    df_q = df_q[(df_q['preclose']>0) & (df_q['open']>0)]

    df_q['code'] = code
    df_q['sname'] = name
    df_q.reset_index(drop=True, inplace=True)

    if len(df_q)>5:
        return df_q
    else:
        return pd.DataFrame()

# 数据获取四级函数 - EMBK
def prepEMBKDdata(code, name, df_status,rank_days=40):

    df_status['code'] = code
    df_status['sname'] = name

    if 'volume' not in df_status.columns:
        df_status['volume'] = df_status['vol']

    df_status['preclose'] = df_status['close'].shift(1)

    df_status['amount'] = df_status['amount']/100000000

    df_status['opct'] = (df_status['open'] / df_status['preclose'] - 1) * 100
    df_status['hpct'] = (df_status['high'] / df_status['preclose'] - 1) * 100
    df_status['lpct'] = (df_status['low'] / df_status['preclose'] - 1) * 100

    df_status['pctR2'] = (df_status["close"]/df_status["close"].shift(2)-1)*100
    df_status['pctR3'] = (df_status["close"]/df_status["close"].shift(3)-1)*100
    df_status['pctR4'] = (df_status["close"]/df_status["close"].shift(4)-1)*100
    df_status['pctR5'] = (df_status["close"]/df_status["close"].shift(5)-1)*100
    df_status['pctR6'] = (df_status["close"]/df_status["close"].shift(6)-1)*100
    df_status['pctR7'] = (df_status["close"]/df_status["close"].shift(7)-1)*100
    df_status['pctR8'] = (df_status["close"]/df_status["close"].shift(8)-1)*100

    df_status['kline'] = df_status.apply(lambda x: 1 if x.close>x.open else 0, axis=1)

    df_status['cp5'] = (df_status["close"] - df_status["close"].rolling(5,min_periods=1).min()) \
                       / (df_status["close"].rolling(5).max() - df_status["close"].rolling(5).min())
    df_status['cp20'] = (df_status["close"] - df_status["close"].rolling(20,min_periods=1).min()) \
                        / (df_status["close"].rolling(20).max() - df_status["close"].rolling(20).min())
    df_status['cp60'] = (df_status["close"] - df_status["close"].rolling(60,min_periods=1).min()) \
                        / (df_status["close"].rolling(60).max() - df_status["close"].rolling(60).min())

    df_status['vt5'] = (df_status["volume"] - df_status["volume"].rolling(5,min_periods=1).min()) \
                       / (df_status["volume"].rolling(5).max() - df_status["volume"].rolling(5).min())
    df_status['vt10'] = (df_status["volume"] - df_status["volume"].rolling(10,min_periods=1).min()) \
                        / (df_status["volume"].rolling(10).max() - df_status["volume"].rolling(10).min())
    df_status['vt20'] = (df_status["volume"] - df_status["volume"].rolling(20,min_periods=1).min()) \
                        / (df_status["volume"].rolling(20).max() - df_status["volume"].rolling(20).min())
    df_status['vt60'] = (df_status["volume"] - df_status["volume"].rolling(60,min_periods=1).min()) \
                        / (df_status["volume"].rolling(60).max() - df_status["volume"].rolling(60).min())
    df_status['vratio5'] = df_status["volume"] / df_status["volume"].rolling(5,min_periods=1).mean()
    df_status['vratio10'] = df_status["volume"] / df_status["volume"].rolling(10,min_periods=1).mean()
    df_status['vratio20'] = df_status["volume"] / df_status["volume"].rolling(20,min_periods=1).mean()
    df_status['vratio60'] = df_status["volume"] / df_status["volume"].rolling(60,min_periods=1).mean()

    df_status['aft1D'] = df_status["close"].shift(-1)/df_status["close"]*100-100
    # df_status['aft2D'] = df_status["close"].shift(-2)/df_status["close"]*100-100
    df_status['aft3D'] = df_status["close"].shift(-3)/df_status["close"]*100-100
    # df_status['aft4D'] = df_status["pctChg"].rolling(4).sum().shift(-4)
    df_status['aft5D'] = df_status["close"].shift(-5)/df_status["close"]*100-100
    # df_status['aft8D'] = df_status["close"].shift(-8)/df_status["close"]*100-100
    # df_status['aft10D'] = df_status["close"].shift(-10)/df_status["close"]*100-100

    df_status['rank'] = df_status['volume'].rolling(rank_days + 1).apply(lambda x: pd.Series(x).rank().iloc[-1])
    df_status['vol_rank'] = 2 * (df_status['rank'] - rank_days - 1) / rank_days + 1
    df_status['bias'] = computeBias(df_status[['close']], 10)
    df_status['biasgood'] = (df_status['bias']<0).map({True: 1, False: 0})
    df_status['biasup'] = df_status['bias'] < df_status['bias'].shift(1)

    df_status['ma5'] = df_status["close"].rolling(5).mean()
    df_status['ma5up'] = (df_status['ma5']>df_status['ma5'].shift(1))
    df_status['ma5delta'] = (df_status['ma5']/df_status['ma5'].shift(1)-1)*100
    df_status.ma5up = df_status.ma5up.map({True: 1, False: 0})
    df_status['abovema5'] = (df_status["close"]>df_status['ma5']).map({True: 1, False: 0})

    df_status['ma10'] = df_status["close"].rolling(10).mean()
    df_status['ma10up'] = df_status['ma10']>df_status['ma10'].shift(1)
    df_status['ma10delta'] = (df_status['ma10'] / df_status['ma10'].shift(1) - 1) * 100
    df_status.ma10up = df_status.ma10up.map({True: 1, False: 0})
    df_status['abovema10'] = (df_status["close"] > df_status['ma10']).map({True: 1, False: 0})

    df_status['kline'] = df_status.apply(lambda x: 1 if x.ma10up and x.close>x.ma10 else 0, axis=1)

    df_status['ma20'] = df_status["close"].rolling(20).mean()
    df_status['ma20up'] = df_status['ma20']>df_status['ma20'].shift(1)
    df_status.ma20up = df_status.ma20up.map({True: 1, False: 0})
    df_status['ma20delta'] = (df_status['ma20'] / df_status['ma20'].shift(1) - 1) * 100
    df_status['abovema20'] = (df_status["close"] > df_status['ma20']).map({True: 1, False: 0})

    df_status['indicator'] = df_status.apply(lambda x: calcIndicator(x.ma5, x.ma10, x.ma20), axis=1)

    df_status['kdj'] = computeKDJv2(df_status.copy())
    df_status['bwidth'], df_status['bupperdelta'] = computeBollv2(df_status.copy())
    # df_status['rsi'] = computeRSI(df_status)
    df_status['rsi'] = ta.RSI(df_status['close'], timeperiod=14)
    # df_status['obv'] = computeOBV(df_status)
    df_status['obv'] = ta.OBV(df_status['close'].values, df_status['volume'].values)
    df_status['obv'] = (df_status['obv']>df_status['obv'].shift(2)).map({True: 1, False: 0})

    df_status['week'] = df_status.apply(lambda x: week_day(x.date), axis=1)
    df_status['c60cls'] = df_status.apply(lambda x: mybin(x.cp60), axis=1)
    df_status['v60cls'] = df_status.apply(lambda x: mybin(x.vt60), axis=1)
    df_status['cv'] = df_status['c60cls'] * 10 + df_status['v60cls']

    return df_status

##################### ML processing #####################

def train2evaluate(df, train_cut_s_date1, train_cut_e_date1, valid_s_date, valid_e_date,  featurelist, model, model_fn, fn_tag, df_dapan, maxselect=5):
    # global sourceFile

    print('\n\n============== starting tran2evaluate model: ', fn_tag, ' =======================')

    print('train_s_date1, train_e_date1,  valid_s_date, valid_e_date')
    print(train_cut_s_date1, train_cut_e_date1,  valid_s_date, valid_e_date)
    print('train2evaluate data shape: ', df.shape)

    transform_start = time.time()

    df_train1, df_valid = dataTransform(df, train_cut_s_date1, train_cut_e_date1, valid_s_date, valid_e_date, featurelist)
    print('-------- After data Transformation for train1 -------')
    print('train_e_date:', train_cut_e_date1, '   valid_s_date:', valid_s_date)
    print('df_train shape:', df_train1.shape, '   df_valid shape:', df_valid.shape)
    print(round(time.time()-transform_start, 0), 'secs to transform data')

    dfo_train1 = pd.DataFrame(df_train1)
    dfo_train1.fillna(0, inplace=True)
    # dfo_train1 = dfo_train1[(dfo_train1.iloc[:,4] < 9.85) & (dfo_train1.iloc[:,4] > -9.9)]
    X_train1 = dfo_train1.iloc[:,6:]
    y_train1 = dfo_train1.iloc[:,5:6].values[:,0]
    print('X_train1 rows: ', len(X_train1), "samples per Tree:", int(len(X_train1)/3*2))

    # if len(df_valid)>0:
    tune_valid = pd.DataFrame(df_valid)
    tune_valid = tune_valid[tune_valid[0] < datetime.datetime.now().strftime('%Y-%m-%d')]
    tune_valid.fillna(0, inplace=True)
    X_valid = tune_valid.iloc[:,6:]
    y_valid = tune_valid.iloc[:,5:6].values[:,0]

    # forest_reg = tune_via_predefinedsplit(X_train1, y_train1, X_valid, y_valid)

    if '_stk_' in fn_tag and 'aft1D' in fn_tag:
        # for stockcount =800
        # param_grid = {'n_estimators': [60], 'max_features': [7,None], 'min_samples_leaf': [1,3], 'min_samples_split': [2, 5], 'max_depth': [9],}
        param_grid = {}
    elif 'aft1D' in fn_tag:
        # param_grid = {'n_estimators': [30, 60], 'max_features': [5, 7], 'min_samples_leaf': [60], 'min_samples_split': [10, 30], 'max_depth': [5, 7],}
        param_grid = {}
    elif '_stk_' in fn_tag and 'aft1W' in fn_tag:
        # param_grid = {'n_estimators': [30, 60], 'max_features': [5, 7], 'min_samples_leaf': [60], 'min_samples_split': [30, 60], 'max_depth': [7, 8],}
        param_grid = {}
    elif 'aft1W' in fn_tag:
        # param_grid = {'n_estimators': [30, 60], 'max_features': [5, 7], 'min_samples_leaf': [60], 'min_samples_split': [30, 60], 'max_depth': [7, 9],}
        param_grid = {}
    elif '_stk_' in fn_tag and 'aft1M' in fn_tag:
        # param_grid = {'n_estimators': [30, 60], 'max_features': [7,None], 'min_samples_leaf': [50,100], 'min_samples_split': [100,200], 'max_depth': [7, 9],}
        param_grid = {}
    elif 'aft1M' in fn_tag:
        # param_grid = {'n_estimators': [30, 60], 'max_features': [5, 7], 'min_samples_leaf': [50,100], 'min_samples_split': [100,200], 'max_depth': [6,7],}
        param_grid = {}
    elif '_stk_' in fn_tag and 'aft1Q' in fn_tag:
        # param_grid = {'n_estimators': [30, 60], 'max_features': [6, 7], 'min_samples_leaf': [50,100], 'min_samples_split': [100,200], 'max_depth': [6, 7],}
        param_grid = {}
    elif 'aft1Q' in fn_tag:
        # param_grid = {'n_estimators': [30, 60], 'max_features': [5, 7], 'max_depth': [5, 7, 9],}
        param_grid = {}
    else:
        print('pls check prog_tag: ', fn_tag)
        param_grid = {}
    print('param_grid:', param_grid)

    if os.path.exists(model_fn):
        # fname = pathlib.Path(model_fn)
        # if datetime.datetime.fromtimestamp(fname.stat().st_mtime).strftime("%Y%m") == \
        #         datetime.datetime.now().strftime('%Y%m'):
        print('\nReusing model files ', model_fn, 'under models: ', fn_tag)
        forest_reg = pickle.load(open(model_fn, 'rb'))
        print("model parameters:", str(forest_reg))
        # else:
        #     print('\nModel exists but not created today, tuning via hypopt...')
        #     forest_reg, param = tune_via_hypopt(X_train1, y_train1, X_valid, y_valid, model, param_grid)
        #     try:
        #         os.rename(model_fn, model_fn.replace(".pickle", "renamedAt_"+datetime.datetime.now().strftime('%Y%m%d')+".pickle"))
        #         model_new_fn = model_fn.replace(".pickle", "renamedAt_"+datetime.datetime.now().strftime('%Y%m%d')+".pickle")
        #         print(model_fn, "已被重命名为", model_new_fn)
        #         pickle.dump(forest_reg, open(model_fn, 'wb'))
        #         print(model_fn, "已重新生成。")
        #         with open(model_new_fn + '.txt', 'w+') as f:
        #             f.write(param)
        #     except:
        #         print(model_fn, "重命名失败！")

    else:
        print('\nModel file not exists, Start training...', 'model name:', model_fn)
        # if fn_tag in ['EMBK2stk_aft1W', 'EMBK2stk_aft1M', 'TdxBK2stk_aft1W', 'TxdBK2stk_aft1M']:
        # forest_reg = trainModel(X_train1, y_train1, model, featurelist)
        forest_reg, param = tune_via_hypopt(X_train1, y_train1, X_valid, y_valid, model, param_grid)
        pickle.dump(forest_reg, open(model_fn, 'wb'))
        with open(model_fn + 'createdAt_' + datetime.datetime.now().strftime('%Y%m%d')+'.txt', 'w+') as f:
            f.write(param)
        # pickle.dump(lin_reg, open('models\\lin_' + fn_tag + '.pickle', 'wb'))

    if len(df_valid)>0:
        dfo_valid = pd.DataFrame(df_valid)
        dfo_valid.fillna(0, inplace=True)
        X_valid = dfo_valid.iloc[:,6:]
        y_valid = dfo_valid.iloc[:,5:6]
        df_validate = dfo_valid.iloc[:,:6]
        df_validate.columns = ['date', 'code', 'sname', 'close', 'pctChg', 'aft1D']
        # dfe_test['aft1D'] = dfe_test['aft1D'].astype(float)
    # if len(y_predict) >0:
        modelEvaluate(forest_reg, (X_valid,y_valid),fn_tag)
        # print(train_result)
        print("\n", "-"*20, "\n Evaluating test data set!")

        df_validate['forest'] = forest_reg.predict(X_valid)
        df_validate['RFclass'] = df_validate.apply(lambda x: round(x.forest/0.1,0), axis=1)
        # dfe_test['lin'] = lin_reg.predict(X_predict)
        # df_validate = dfe_test[dfe_test['date']<pred_cut_e_date]
        # df_predict = dfe_test[dfe_test['date']>=pred_cut_e_date]

        df_validate['source'] = fn_tag

        print('\n', 'df_validate rows before removing duplicates: ', len(df_validate))
        df_validate = df_validate.drop_duplicates(subset=['code', 'date'], keep='last')
        print('df_validate rows after removing duplicates: ', len(df_validate))
        print('df_validate shape:', df_validate.shape)

        rf_best_threshold, df_validate = analysisTest(df_validate, fn_tag, df_dapan, maxselect)
        # rf_best_threshold = analysisTest(df_validate,rf_buy, fn_tag)
        # print('best RF threshold', rf_best_threshold)

        if rf_best_threshold == 9999:
            print(fn_tag, 'model no good, prediction skipped.')
            if '_stk_' in fn_tag:
                return df_validate
            else:
                return pd.DataFrame()
        else:
            df_validate['flag'] = df_validate.apply(lambda x: 1 if x.ranktopN ==1 else 0, axis=1)
            df_validate.to_csv(output_fn_prefix + fn_tag + '_all_validation.csv', encoding='gbk', index=False)
            tmp = df_validate[df_validate['flag'] == 1]

            if len(tmp) > 0:
                print(fn_tag, '+++++++ predict data length:', len(tmp), '+++++++')

                tmp.sort_values(by='date', inplace=True)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    tmp = tmp.tail(30)
                    print(tmp[['date', 'code', 'sname', 'close', 'pctChg', 'forest','RFclass', 'aft1D']])
            else:
                print(fn_tag, 'XXXXXX validation processed, but BK selected from ', valid_s_date)

        return df_validate

def dataTransform(data,trainS, trainE, predictS,predictE, features):

    cat_used = False
    if isinstance(features[-1], list):
        cat_features = features[-1]
        fix_features = features[:-1]
        cat_used = True
    else:
        fix_features = features
        # if 'cv' in cat_features:
        #     data['cv'] = data['c60cls'] * 10 + data['v60cls']

    # print("\ndata before transmation:", "\n", data.info())
    # data = data.dropna(how="any")
    # print("data rows after dropna:", len(data))

    train_set = data[(data['date']>=trainS) & (data['date']<=trainE)]
    predict_set = data[(data['date']>=predictS)]
    # predict_set = data[(data['date']>=predictS) & (data['date']<=predictE)]

    fix_pipeline = Pipeline([
        ('selector', DataFrameSelector(fix_features)), ])

    if cat_used == True:
        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_features)),
            ('cat_encoder', OneHotEncoder(sparse=False)), ])  # LabelEncoder, CategoricalEncoder???

        full_pipeline = FeatureUnion(transformer_list=[
            ("fix_pipeline", fix_pipeline),
            ("cat_pipeline", cat_pipeline), ])
    else:
        full_pipeline = FeatureUnion(transformer_list=[
            ("fix_pipeline", fix_pipeline),])
            # ("num_pipeline", num_pipeline),
            # ("cat_pipeline", cat_pipeline), ])
    train = full_pipeline.fit_transform(train_set)
    test = full_pipeline.transform(predict_set) if len(predict_set)>0 else pd.DataFrame()

    return train, test

# def ms_autoML(X_train,y_train, X_predict, dfe_test):
#     # ############## Microsoft flaml ######################
#     automl = AutoML()
#     # Specify automl goal and constraint
#     automl_settings = {
#         "time_budget": 400,  # in seconds
#         "metric": 'rmse',
#         "task": 'regression',
#         "log_file_name": "outputQ\\AutoML.log",
#         # "estimator_list":["rf","lgbm","xgboost", "catboost"],
#         "estimator_list": ["lgbm"],
#     }
#     automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
#     # Export the best model
#     # LGBMRegressor(colsample_bytree=0.47664134378616263,
#     #               learning_rate=0.13017784899354856, max_bin=15,
#     #               min_child_samples=10, n_estimators=332, num_leaves=63,
#     #               reg_alpha=0.0009765625, reg_lambda=0.0024520117287225355,
#     #               verbose=-1)
#     # mean absolute error value : 2.1596833394388275
#     # mean squared error : 9.666887356957819
#     # r2 score : -0.024655100108934347
#     #----- below overfit para
#     # LGBMRegressor(colsample_bytree=0.5449675270308239,
#     #               learning_rate=0.010350220894132149, max_bin=1023,
#     #               min_child_samples=8, n_estimators=1931, num_leaves=2807,
#     #               reg_alpha=0.14086685048562353, reg_lambda=0.0010117170961620916,
#     #               verbose=-1)
#     #########################################
#     # RandomForestRegressor(max_features=0.1750617153084237, max_leaf_nodes=32767,
#     #                       n_estimators=573, n_jobs=-1)
#     # mean absolute error value : 2.1306983663888817
#     # mean squared error : 9.531515855892428
#     # r2 score : -0.01030621055906078
#     # good score, but bad with predict>1, good@0.3-0.9  output0.365x41000
#     #########################################
#     print(automl.model)
#     with open('automl.pkl', 'wb') as f:
#         pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
#     # Predict
#     prediction = automl.predict(X_predict)
#     dfe_test['forest'] = prediction
#
#     dfe_test.dropna(inplace=True)
#
#     # modelEvaluate(forest_reg, (X_predict, y_predict))
#
#     df_validate = dfe_test[dfe_test['date'] < pred_cut_e_date]
#     df_predict = dfe_test[dfe_test['date'] >= pred_cut_e_date]
#     analysisTest(df_validate)
#     exit(0)
#
#     ############## Microsoft flaml ######################
#     # print('accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', prediction, y_predict))
#     # print('roc_auc', '=', 1 - sklearn_metric_loss_score('roc_auc', prediction, y_predict))
#     # print('log_loss', '=', sklearn_metric_loss_score('log_loss', prediction, y_predict))
#     # print('max error value :', max_error(y_predict, prediction))
#     # print('mean absolute error value :', mean_absolute_error(y_predict, prediction))
#     # print('mean squared error :', mean_squared_error(y_predict, prediction))
#     # print("mean squared log error :", mean_squared_log_error(y_predict, prediction))
#     # print("r2 score :", r2_score(y_predict, prediction))
#     # <flaml.model.LGBMEstimator object at 0x000001ACB4CBC160>
#     # mean absolute error value : 2.1991771756929563
#     # mean squared error : 10.183492805921297
#     # r2 score : -0.017963039309168094
#     # <flaml.model.RandomForestEstimator object at 0x000001DD8AD08748>
#     # mean absolute error value : 2.4780595634908513
#     # mean squared error : 13.19306318970577
#     # r2 score : -0.31560362271940146

def trainModel(X_train,y_train, rf_reg, TdxBKfeatures):
    # global train_result, stamp, sourceFile, TdxBKfeatures

    print('train set shape:', X_train.shape)

    # templist =  featurelist + ['week1', 'week2', 'week3', 'week4', 'week5']
    train_features = TdxBKfeatures[6:]

    # v0.9 original
    # forest_reg = RandomForestRegressor(n_estimators=93, n_jobs=-1, max_features=17, min_samples_split=7,
    #                                    min_samples_leaf=6, max_depth=15, random_state=42)
    # forest_reg = RandomForestRegressor(n_estimators=187, n_jobs=-1, max_features=10,  # min_samples_split=7,
    #                                    min_samples_leaf=100, max_depth=15, random_state=42)
    # forest_reg = RandomForestRegressor(n_estimators=93, n_jobs=-1, max_features=17, min_samples_leaf=9,
    #                                    min_samples_split=40, max_depth=15, random_state=42)
    # 2021-11-22
    # forest_reg = RandomForestRegressor(n_estimators=550, n_jobs=10, max_features=9, min_samples_leaf=485,
    #                                    min_samples_split=291, max_depth=7, random_state=42,oob_score = True,)

    model_start = time.time()
    print("\n模型类别 ", rf_reg.__class__.__name__)
    print(rf_reg)
    rf_reg.fit(X_train, y_train)
    print('time cost: ', round(time.time() - model_start, 2))


    print("feature importance list length: ", len(list(rf_reg.feature_importances_)))
    print(sorted(zip(roundTo3(list(rf_reg.feature_importances_)), train_features), reverse=True))

    return rf_reg   #, lin_reg

def modelEvaluate(estimator, testdata, modeltag):

    if isinstance(testdata, tuple):
        X_test, y_test = testdata
    else:
        return (0, 0, 0, 0)

    model_start = time.time()

    print("\n模型类别 ", estimator.__class__.__name__)
    print("estimator.score(X_test, y_test) 默认评估值为：", estimator.score(X_test, y_test))

    test_predictions = estimator.predict(X_test)
    predictused = time.time() - model_start
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    valid_result = pd.DataFrame({'modeltag':[modeltag], 'data': ['Test'], 'model': [estimator.__class__.__name__], 'rmse': [test_rmse],
                                        'mae': [test_mae], 'r2': [test_r2], 'timeused': [predictused]})
    print('model evaluate result\n', valid_result)
    end = time.time()
    print('time cost: ', round(end - model_start, 2))

    return (test_rmse, test_mae)

def model_tune_curve(X_train,y_train, model):

    param_range = np.arange(5, 100, 5)
    train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="n_estimators", param_range=param_range,
                                                 cv=4, scoring="neg_mean_squared_error", n_jobs=8)    # neg_mean_squared_error  r2
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.subplots(1, figsize=(7, 7))
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    plt.title("Validation Curve With Random Forest")
    plt.xlabel("Number Of Trees")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    # plt.show()
    plt.savefig('EMBK2stk_tune_aft1D_trees.png')
    plt.close()


    param_range = np.arange(1, 30, 2)
    train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="max_features",
                                                 param_range=param_range,
                                                 cv=4, scoring="neg_mean_squared_error", n_jobs=8)  # neg_mean_squared_error, r2
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.subplots(1, figsize=(7, 7))
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    plt.title("Validation Curve With Random Forest")
    plt.xlabel("max_features")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    # plt.show()
    plt.savefig('EMBK2stk_tune_aft1D_max_features_.png')
    plt.close()

def calcScore(id, estimator, X_test, y_test):
    train_predictions = estimator.predict(X_test)
    train_mse = mean_squared_error(y_test, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_test, train_predictions)
    train_r2 = r2_score(y_test, train_predictions)
    return {'data': id, 'model': estimator.__class__.__name__, 'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2}

def tune_example(X,y,model):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_validate
    from sklearn import metrics
    # from sklearn import cross_validation, metrics

    # train = pd.read_csv('train_modified.csv')
    # target = 'Disbursed'  # Disbursed的值就是二元分类的输出
    # IDcol = 'ID'
    # train['Disbursed'].value_counts()
    # 0    19680
    # 1      320
    # Name: Disbursed, dtype: int64
    # 不管任何参数，都用默认的，我们拟合下数据看看：
    rf0 = RandomForestRegressor(oob_score=True, random_state=10)
    rf0.fit(X, y)
    print(rf0.oob_score_)
    # y_predprob = rf0.predict_proba(X)[:, 1]
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

    # 我们首先对n_estimators进行网格搜索：
    # param_test1 = {'n_estimators': range(10, 71, 10)}
    # gsearch1 = GridSearchCV(estimator=RandomForestRegressor(min_samples_split=100,
    #                                                          min_samples_leaf=20, max_depth=8, max_features='sqrt',
    #                                                          random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', cv=5)
    # gsearch1.fit(X, y)
    # # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    # print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
    param_test1 = {'n_estimators': range(3, 150, 7)}
    gsearch1 = GridSearchCV(estimator=rf0, param_grid=param_test1, scoring='r2', cv=3)
    gsearch1.fit(X, y)
    print(rf0.oob_score_)
    print('gsearch1.cv_results_', gsearch1.cv_results_)
    print('gsearch1.best_params_',gsearch1.best_params_)
    print('gsearch1.best_score_', gsearch1.best_score_)
    print('gsearch1.best_estimator_', gsearch1.best_estimator_)

    # 接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
    # param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
    # gsearch2 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=60,
    #                                                          min_samples_leaf=20, max_features='sqrt', oob_score=True,
    #                                                          random_state=10),
    #                         param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    # gsearch2.fit(X, y)
    # print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
    rf0.n_estimators = gsearch1.best_params_['n_estimators']
    param_test2 = {'max_depth': range(3, 21, 2), 'min_samples_split': range(20, 201, 20)}
    gsearch2 = GridSearchCV(estimator=rf0, param_grid=param_test2, scoring='r2', cv=3)
    gsearch2.fit(X, y)
    print(rf0.oob_score_)
    print('gsearch2.cv_results_', gsearch2.cv_results_)
    print('gsearch2.best_params_',gsearch2.best_params_)
    print('gsearch2.best_score_', gsearch2.best_score_)
    print('gsearch2.best_estimator_', gsearch2.best_estimator_)


    # 对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再# 划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
    # param_test3 = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
    # gsearch3 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=60, max_depth=13,
    #                                                          max_features='sqrt', oob_score=True, random_state=10),
    #                         param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    rf0.max_depth = gsearch2.best_params_['max_depth']
    rf0.min_samples_split = gsearch2.best_params_['min_samples_split']
    param_test3 = {'min_samples_split': range(gsearch2.best_params_['min_samples_split']-50, gsearch2.best_params_['min_samples_split']+50, 20), 'min_samples_leaf': range(10, 150, 20)}
    gsearch3 = GridSearchCV(estimator=rf0,   param_grid=param_test3, scoring='r2', cv=3)
    gsearch3.fit(X, y)
    print('gsearch3.best_estimator_', gsearch3.best_estimator_)
    print('gsearch3.cv_results_', gsearch3.cv_results_)
    print('gsearch3.best_params_', gsearch3.best_params_)
    print('gsearch3.best_score_', gsearch3.best_score_)

    # 最后我们再对最大特征数max_features做调参:
    # param_test4 = {'max_features': range(3, 11, 2)}
    # gsearch4 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=60, max_depth=13, min_samples_split=120,
    #                                                          min_samples_leaf=20, oob_score=True, random_state=10),
    #                         param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
    rf0.min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
    rf0.min_samples_split = gsearch3.best_params_['min_samples_split']
    param_test4 = {'max_features': range(3, 46, 3)}
    gsearch4 = GridSearchCV(estimator=rf0, param_grid=param_test4, scoring='r2', cv=3)
    gsearch4.fit(X, y)
    print('gsearch4.cv_results_', gsearch4.cv_results_)
    print('gsearch4.best_params_', gsearch4.best_params_)
    print('gsearch4.best_score_', gsearch4.best_score_)


    # 用我们搜索到的最佳参数，我们再看看最终的模型拟合：
    rf2 = gsearch4.best_estimator_
    rf2.fit(X, y)
    print(rf2.oob_score_)
    print('gsearch4.best_estimator_', gsearch4.best_estimator_)

    return gsearch4.best_estimator_

def tune_via_hypopt(X_train,y_train,X_val,y_val, base_model, param_grid):
    # from hypopt import GridSearch
    from custopt import GridSearch

    if len(param_grid) > 0:
        print('\ntune_via_hyopt ...... ')

        # Grid-search all parameter combinations using a validation set.
        gs = GridSearch(model=RandomForestRegressor(oob_score=True, random_state=42, n_jobs=8, verbose=1), param_grid=param_grid, parallelize=False)
        _, para = gs.fit(X_train, y_train, X_val, y_val, scoring='custom_neg_mean_squared_error')
        hyopt_result = calcScore('hyopt_result', gs, X_val, y_val)
        print('best estimator:', gs.best_estimator_)
        print('hyopt_result\n',pd.DataFrame([hyopt_result]))
        print('hyopt_parameters\n', gs.params)
        print('hyopt_scores\n', gs.scores)
        print('hyopt_best_params\n', para)
        # Compare with model without hyperopt
        _ = base_model.fit(X_train, y_train)
        base_result = calcScore('base_result', base_model, X_val, y_val)
        print('\nbase estimator:', base_model)
        print('base_result\n', pd.DataFrame([base_result]))
        return gs, json.dumps(para)
    else:
        print('\nparam_grid is blank, using fixed parameters...')
        _ = base_model.fit(X_train, y_train)
        para = str(base_model)
        base_result = calcScore('base_result', base_model, X_val, y_val)
        print('\nbase estimator:', base_model)
        print('base_result\n', pd.DataFrame([base_result]))
        return base_model, para

def tune_via_predefinedsplit(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    from sklearn.model_selection import cross_validate
    from sklearn import metrics
    # from sklearn import cross_validation, metrics

    tune_result = pd.DataFrame()

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, np.array(y_test)[:,0]])

    test_fold = [-1] * len(X_train) + [0] * len(X_test)
    ps = PredefinedSplit(test_fold)
    print('ps.get_n_splits()', ps.get_n_splits())

    # 不管任何参数，都用默认的，我们拟合下数据看看：
    rf0 = RandomForestRegressor(oob_score=True, random_state=10, n_jobs=8, )
    rf0.fit(X_train, y_train)
    result = calcScore('0allDefault', rf0, X_test, y_test)
    tune_result = pd.DataFrame([result])
    print(tune_result)

    # # 我们首先对n_estimators进行网格搜索：
    # param_test1 = {'n_estimators': range(30, 150, 10)}
    # gsearch1 = GridSearchCV(estimator=rf0, param_grid=param_test1, scoring='r2', cv=ps)
    # gsearch1.fit(X, y)
    # print(rf0.oob_score_)
    # print('gsearch1.cv_results_', gsearch1.cv_results_)
    # print('gsearch1.best_params_',gsearch1.best_params_)
    # print('gsearch1.best_score_', gsearch1.best_score_)
    # print('gsearch1.best_estimator_', gsearch1.best_estimator_)
    # rf0.n_estimators = gsearch1.best_params_['n_estimators']
    # rf0.fit(X_train, y_train)
    # result = calcScore('1best_n_estimators', rf0, X_test, y_test)
    # tune_result = tune_result.append(result, ignore_index=True)
    #
    # # 接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
    # rf0.n_estimators = gsearch1.best_params_['n_estimators']
    # param_test2 = {'max_depth': range(5, 18, 2), 'min_samples_split': range(20, 201, 20)}
    # gsearch2 = GridSearchCV(estimator=rf0, param_grid=param_test2, scoring='r2', cv=ps)
    # gsearch2.fit(X, y)
    # print(rf0.oob_score_)
    # print('gsearch2.cv_results_', gsearch2.cv_results_)
    # print('gsearch2.best_params_',gsearch2.best_params_)
    # print('gsearch2.best_score_', gsearch2.best_score_)
    # print('gsearch2.best_estimator_', gsearch2.best_estimator_)
    # rf0.max_depth = gsearch2.best_params_['max_depth']
    # rf0.min_samples_split = gsearch2.best_params_['min_samples_split']
    # rf0.fit(X_train, y_train)
    # result = calcScore('2best_maxdepth_minsplit', rf0, X_test, y_test)
    # tune_result = tune_result.append(result, ignore_index=True)
    #
    # # 对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再# 划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
    # rf0.max_depth = gsearch2.best_params_['max_depth']
    # rf0.min_samples_split = gsearch2.best_params_['min_samples_split']
    # param_test3 = {'min_samples_split': range(gsearch2.best_params_['min_samples_split']-50, gsearch2.best_params_['min_samples_split']+50, 20), 'min_samples_leaf': range(10, 150, 20)}
    # gsearch3 = GridSearchCV(estimator=rf0,   param_grid=param_test3, scoring='r2', cv=ps)
    # gsearch3.fit(X, y)
    # print('gsearch3.best_estimator_', gsearch3.best_estimator_)
    # print('gsearch3.cv_results_', gsearch3.cv_results_)
    # print('gsearch3.best_params_', gsearch3.best_params_)
    # print('gsearch3.best_score_', gsearch3.best_score_)
    # rf0.min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
    # rf0.min_samples_split = gsearch3.best_params_['min_samples_split']
    # rf0.fit(X_train, y_train)
    # result = calcScore('3best_minsplit_minleaf', rf0, X_test, y_test)
    # tune_result = tune_result.append(result, ignore_index=True)
    #
    # # 最后我们再对最大特征数max_features做调参:
    # rf0.min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
    # rf0.min_samples_split = gsearch3.best_params_['min_samples_split']
    # param_test4 = {'max_features': range(3, 46, 3)}
    # gsearch4 = GridSearchCV(estimator=rf0, param_grid=param_test4, scoring='r2', cv=ps)
    # gsearch4.fit(X, y)
    # print('gsearch4.cv_results_', gsearch4.cv_results_)
    # print('gsearch4.best_params_', gsearch4.best_params_)
    # print('gsearch4.best_score_', gsearch4.best_score_)
    # rf0.max_features = gsearch3.best_params_['max_features']

    rf1 = RandomForestRegressor(n_estimators=140, oob_score=True, random_state=10, n_jobs=8, max_features=9, min_samples_leaf=10, min_samples_split=30, max_depth=5)
    rf1.fit(X_train, y_train)
    result = calcScore('4best_maxfeatures', rf1, X_test, y_test)
    tune_result = tune_result.append(result, ignore_index=True)

    print('\ntune_result:\n', tune_result)

    return rf1

def train2search(X_train,y_train, model):

    # model_tune_curve(X_train, y_train, model)

    # rfr = tune_example(X_train, y_train, model)

    param_distribs = {
        'n_estimators': [30,60],
        # 'n_estimators': randint(low=100, high=400),
        # 'max_features': randint(low=5, high=11),
        'max_features': [7,9,None],
        # 'min_samples_leaf': 485,
        # 'min_samples_leaf': randint(low=100, high=700),
        'max_depth': [7, 8,9,10,11],
        # 'max_depth': 7 is the best, 10 is overfit.
        # 'min_samples_split': 291,
        # 'min_samples_split': randint(low=100, high=700),
    }

    # forestreg = RandomForestRegressor(n_estimators=550, n_jobs=10, max_features=9, min_samples_leaf=485,
    #                                    min_samples_split=291, max_depth=7, random_state=42, oob_score=True, verbose=1)

    EMBK1W_model = RandomForestRegressor(n_estimators=30, n_jobs=8, max_depth=9, max_features=9, # min_samples_leaf=100,min_samples_split=300,
                                     random_state=42, oob_score=True, verbose=1)

    EMBK1M_model = RandomForestRegressor(n_estimators=60, n_jobs=8, max_depth=11, max_features=9, # min_samples_leaf=100, min_samples_split=300,
                                     random_state=42, oob_score=True, )

    TdxBK1M_model = RandomForestRegressor(n_estimators=60, n_jobs=8, max_depth=11, max_features=None, # min_samples_leaf=100, min_samples_split=300,
                                     random_state=42, oob_score=True, )

    # STK1D_RF = RandomForestRegressor(n_estimators=30, n_jobs=8, max_depth=11, max_features=6, min_samples_leaf=100,
    #                                  min_samples_split=200,random_state=42,oob_score = True,)

    rnd_search = RandomizedSearchCV(TdxBK1M_model, param_distributions=param_distribs,
                                    n_iter=12, cv=3, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(X_train, y_train)
    print(rnd_search.best_params_)
    # 2.7359401337958698 {'n_estimators': 30}
    # 2.733083929953496 {'n_estimators': 60}
    # mean absolute error value : 2.1240878745523384
    # mean squared error : 9.423321433961704
    # r2 score : 0.002767927407277382
    print(rnd_search.best_estimator_)

    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    feature_importances = roundTo3(list(rnd_search.best_estimator_.feature_importances_))

    # templist =  ['turn', 'pctChg', 'opct', 'hpct', 'lpct',
    #              'pctR2', 'pctR3', 'pctR4', 'pctR5', 'pctR6', 'pctR7', 'pctR8',
    #              'kline', 'cp5', 'cp20', 'cp60', 'vt5', 'vt10', 'vt20', 'vt60',
    #              'vratio5', 'vratio10', 'vratio20', 'vratio60', 'indicator', 'vol_rank',
    #              'bias', 'biasgood', 'biasup', 'ma5up', 'ma10up', 'ma20up', 'abovema5', 'abovema10', 'abovema20',
    #              'kdj', 'bwidth', 'bupperdelta', 'rsi', 'obv',
    #              'dpamtrank', 'dpenv', 'dpbias', 'dpbiasup',
    #              'MApctChg', 'MApctR2', 'MApctR5', 'MApctR8', 'MAma5up', 'MAma10up',
    #              'MAlpct', 'MAopct', 'MAhpct', 'MAturn', 'MAbias', 'MAbiasup', 'MAbwidth', 'MAvol_rank',
    #              'MArsi', 'MAkdj', 'MAbupperdelta', 'MAobv', 'MAvt60', 'MAvt10', 'MAcp5', 'MAcp20', 'MAcp60',
    #              'gappctChg', 'gappctR2', 'gappctR5', 'gappctR8', 'gapma5up', 'gapma10up',
    #              'gaplpct', 'gapopct', 'gaphpct', 'gapturn', 'gapbias', 'gapbiasup', 'gapbwidth', 'gapvol_rank',
    #              'gaprsi', 'gapkdj', 'gapbupperdelta', 'gapobv', 'gapvt60', 'gapvt10', 'gapcp5', 'gapcp20','gapcp60',
    #              'week1', 'week2', 'week3', 'week4', 'week5']

    # {'max_depth': 34, 'max_features': 18, 'min_samples_leaf': 47, 'min_samples_split': 44, 'n_estimators': 74}
    # RandomForestRegressor(max_depth=34, max_features=18, min_samples_leaf=47,
    #                       min_samples_split=44, n_estimators=74, n_jobs=10,
    #                       oob_score=True, random_state=42, verbose=1)
    # 2.714493701358707 {'max_depth': 11, 'max_features': 8, 'min_samples_leaf': 29, 'min_samples_split': 15, 'n_estimators': 126}
    # 2.689117168754888 {'max_depth': 12, 'max_features': 17, 'min_samples_leaf': 21, 'min_samples_split': 39, 'n_estimators': 141}
    # 2.5324483548003265 {'max_depth': 23, 'max_features': 11, 'min_samples_leaf': 11, 'min_samples_split': 11, 'n_estimators': 107}
    # 2.5366516310488003 {'max_depth': 25, 'max_features': 8, 'min_samples_leaf': 40, 'min_samples_split': 24, 'n_estimators': 72}
    # 2.828950195168403 {'max_depth': 6, 'max_features': 12, 'min_samples_leaf': 44, 'min_samples_split': 30, 'n_estimators': 57}
    # 2.832001088753741 {'max_depth': 6, 'max_features': 16, 'min_samples_leaf': 21, 'min_samples_split': 33, 'n_estimators': 77}
    # 2.528104655754488 {'max_depth': 26, 'max_features': 17, 'min_samples_leaf': 44, 'min_samples_split': 25, 'n_estimators': 68}
    # 2.5202439442129085 {'max_depth': 31, 'max_features': 15, 'min_samples_leaf': 42, 'min_samples_split': 28, 'n_estimators': 34}
    # 2.5153445809664974 {'max_depth': 34, 'max_features': 18, 'min_samples_leaf': 47, 'min_samples_split': 44, 'n_estimators': 74}
    # 2.5317227059667804 {'max_depth': 24, 'max_features': 13, 'min_samples_leaf': 3, 'min_samples_split': 37, 'n_estimators': 70}
    # [('0.0538', 'MApctR8'), ('0.0535', 'obv'), ('0.0455', 'MApctR5'), ('0.0435', 'MAvol_rank'), ('0.0426', 'dpamtrank'), ('0.0413', 'MAma5up'), ('0.0407', 'turn'),
    # ('0.0383', 'MAma10up'), ('0.0382', 'MApctR2'), ('0.0348', 'MAbias'), ('0.0345', 'dpenv'), ('0.034', 'hpct'), ('0.0326', 'MAcp20'), ('0.0317', 'MApctChg'),
    # ('0.0314', 'MAcp5'), ('0.0311', 'MAcp60'), ('0.0289', 'dpbiasup'), ('0.0278', 'vol_rank'), ('0.0251', 'opct'), ('0.0208', 'cp20'), ('0.0195', 'lpct'), ('0.0177', 'pctChg'),
    # ('0.0172', 'pctR2'), ('0.0164', 'pctR7'), ('0.0156', 'pctR3'), ('0.0155', 'vratio20'), ('0.0147', 'pctR6'), ('0.0142', 'pctR5'), ('0.0141', 'pctR4'), ('0.0131', 'week4'),
    # ('0.0124', 'vratio10'), ('0.0114', 'cp5'), ('0.0096', 'vt20'), ('0.0093', 'vratio5'), ('0.0091', 'vt60'), ('0.0071', 'vt10'), ('0.0069', 'kline'), ('0.0062', 'indicator'),
    # ('0.0059', 'vt5'), ('0.0047', 'cp60'), ('0.004', 'week3'), ('0.004', 'week2'), ('0.004', 'dpbias'), ('0.0039', 'MAma20up'), ('0.003', 'week1'), ('0.0025', 'ma10up'),
    # ('0.0022', 'vratio60'), ('0.0012', 'bias'), ('0.0011', 'rsi'), ('0.0011', 'ma5up'), ('0.0008', 'pctR8'), ('0.0006', 'boll'), ('0.0006', 'biasup'), ('0.0001', 'ma20up'), ('0.0', 'kdj')]

    # train_features = templist
    # print(sorted(zip(feature_importances, train_features), reverse=True))

    return rnd_search.best_estimator_

##################### data analysis #####################

def analysisTest(new_pd, fn_tag, df_dapan, topn):
    # global predStartDate, stamp,  sourceFile, stkrate,etfrate

    # new_pd['RFclass'] = new_pd.apply(lambda x: round(x.forest/0.1,0), axis=1)
    # new_pd.loc[new_pd['pctChg']>=9.9, 'zdt'] = 1
    # new_pd.loc[new_pd['pctChg']<=-9.9, 'zdt'] = 1
    # new_pd['lin_flag'] = new_pd.apply(lambda x: 1 if x.lin >= buy_lin else 0, axis=1)
    # new_pd['all'] = new_pd.apply(lambda x: x['RF_flag'] + x['lin_flag'], axis=1)
    # new_pd['ttl'] = new_pd.apply(lambda x: x.forest + x.lin, axis=1)
    # new_pd.to_csv("..\\daily_output\\TdxBK2stk_Qtr_validate_" + fn_tag + '_'+ stamp + "_v0.1.csv", index=False, encoding="gbk", float_format='%.3f')
    # new_pd = new_pd[(new_pd["pctChg"] < 9.85) & (new_pd["pctChg"] > -9.85)]

    print("\nMaxselect:", topn)
    rf_best_threshold = exploreBestThreshold(new_pd,fn_tag, topn)

    if rf_best_threshold < new_pd['RFclass'].max():
        print("\n==== rf_best_threshold" , rf_best_threshold)

        new_pd['RF_flag'] = new_pd.apply(lambda x: 1 if x.RFclass >= rf_best_threshold else 0, axis=1)
        # new_pd['RF_flag'] = new_pd.apply(lambda x: 1 if x.forest >= rf_threshold else 0, axis=1)

        if fn_tag[-1].lower() == "d":
            # print("save full valiation set to ", output_fn_prefix + fn_tag + "_validate-full.csv")
            # new_pd.to_csv(output_fn_prefix + fn_tag + "_validate-full.csv", index=False, encoding="gbk")
            pass

        if fn_tag[-1] == "Q":
            ylimit = 100
        elif fn_tag[-1] == "M":
            ylimit = 50
        elif fn_tag[-1] == "W":
            ylimit = 25
        elif fn_tag[-1] == "D":
            ylimit = 15
        else:
            print('prog_tag', fn_tag,  'last char NOT Q/M/W/D pls check')
            ylimit = 50
        plotTestPctchg2Aft1d(output_fn_prefix + fn_tag + "_validate_pctChg2Actual_" + ".png", new_pd, ylimit)
        plotTestPred2Aft1d(output_fn_prefix + fn_tag + "_validate_Pred2Actual_" + ".png", new_pd, ylimit)

        df_topN = pd.DataFrame()
        for day in new_pd.date.unique():
            temp = new_pd[(new_pd["date"] == day)]
            temp['rank'] = temp['forest'].rank(ascending=False)
            # temp['rankrev'] = temp['forest'].rank(ascending=True)
            temp.loc[(temp['rank'] <= topn), 'puretopN'] = 1
            temp.loc[(temp['rank'] <= topn) & (temp['RFclass'] >= rf_best_threshold), 'ranktopN'] = 1
            temp.loc[(temp['rank'] > topn) & (temp['rank'] <= topn*2) & (temp['RFclass'] >= rf_best_threshold), 'rankN_2N'] = 1
            # if len(temp[(temp['rank']>5) & (temp['RFclass']>=rf_best_threshold)])>0:
            if len(temp[temp['RFclass'] >= rf_best_threshold]) >= 5:
                temp['cntmore'] = 1
            else:
                temp['cntmore'] = 0
            df_topN = pd.concat([df_topN, temp])
        df_topN.fillna(0, inplace=True)
        df_topN.to_csv(output_fn_prefix + fn_tag + "_rank_info_" + fn_tag + ".csv", index=False, encoding="gbk", float_format='%.3f')

        print("\nMax_selection:", topn)
        print('\nvalidation rows:', len(df_topN), 'average aft1d:', round(df_topN['aft1D'].mean(), 2))
        print('puretopN rows:', len(df_topN[(df_topN['puretopN'] == 1)]), 'average aft1d:',
              round(df_topN[df_topN['puretopN'] == 1]['aft1D'].mean(), 2))
        print('RFclass rows:', len(df_topN[(df_topN['RFclass'] >= rf_best_threshold)]), 'average aft1d:',
              round(df_topN[df_topN['RFclass'] >= rf_best_threshold]['aft1D'].mean(), 2))
        print('RFclass&topN rows:',
              len(df_topN[(df_topN['ranktopN'] == 1) & (df_topN['RFclass'] >= rf_best_threshold)]), 'average aft1d:',
              round(df_topN[(df_topN['ranktopN'] == 1) & (df_topN['RFclass'] >= rf_best_threshold)]['aft1D'].mean(), 2))
        print('RFclass&rankN_2N rows:',
              len(df_topN[(df_topN['rankN_2N'] == 1) & (df_topN['RFclass'] >= rf_best_threshold)]), 'average aft1d:',
              round(df_topN[(df_topN['rankN_2N'] == 1) & (df_topN['RFclass'] >= rf_best_threshold)]['aft1D'].mean(), 2))


        new_pd_buy = df_topN[df_topN["ranktopN"]==1]
        pd_buy = new_pd_buy.groupby('date')['aft1D'].agg(['mean','count','sum']).reset_index()
        # pd_sell = new_pd_sell.groupby('date')['aft1D'].agg(['mean','count','sum']).reset_index()

        pd_buy['cnt'] = pd_buy['count']
        pd_buy.cnt[pd_buy.cnt>topn] = 5
        pd_buy['ceiled'] = pd_buy['mean'] * pd_buy['cnt']

        if 'aft10' in fn_tag.lower():
            pd_buy['capitalpct'] = pd_buy.apply(lambda x: x.ceiled/1000/topn - stkrate*x.cnt/10/topn, axis=1)
        elif 'aft5' in fn_tag.lower():
            pd_buy['capitalpct'] = pd_buy.apply(lambda x: x.ceiled/500/topn - stkrate*x.cnt/5/topn, axis=1)
        elif 'aft1' in fn_tag.lower():
            pd_buy['capitalpct'] = pd_buy.apply(lambda x: x.ceiled/100/topn - stkrate*x.cnt/topn, axis=1)
        else:
            print('invalid fn_tag', fn_tag)

        pd_buy['capital'] = (pd_buy['capitalpct'] + 1).cumprod()

        print("\n" + fn_tag + " buy list" + "\n", pd_buy)
        # print("\n"+ fn_tag+' - '+ predStartDate + " total buy gain :" , round(pd_buy["ceiled"].sum(),2))
        print("-"*40)

        # df = pd.concat([new_pd_buy,new_pd_sell])
        df = new_pd_buy.copy()
        plt.scatter(df["forest"], df["aft1D"], s=1,color='r', alpha=0.5, label='RF')
        # plt.scatter(df["lin"], df["aft1D"],s=9,color='b',alpha=0.5, label='xgbr')
        plt.legend()
        plt.grid()
        plt.ylim((-20, 20))
        plt.xlabel('prediction')
        plt.ylabel('aft1D')
        plt.title(fn_tag)
        plt.savefig(output_fn_prefix + fn_tag + "_validate_report_" + ".png")
        plt.close()

        df_dapan = df_dapan[df_dapan['date'] >= pd_buy['date'][0]]
        pd_buy = pd.merge(pd_buy, df_dapan[['date', 'szclose', 'shclose']], how='right', on='date')
        pd_buy['capital'] = pd_buy['capital'].ffill()
        pd_buy['szclose'] = pd_buy['szclose'] / list(pd_buy['szclose'])[0]
        pd_buy['shclose'] = pd_buy['shclose'] / list(pd_buy['shclose'])[0]

        validperiod = list(pd_buy.date.unique())[0]+'_'+list(pd_buy.date.unique())[-1]
        pd_buy['date'] = pd.to_datetime(pd_buy['date'])
        validcnt = (pd.to_datetime(pd_buy.date.unique()[-1])-pd.to_datetime(pd_buy.date.unique()[0])).days
        pd_buy['date'] = pd.to_datetime(pd_buy['date'])

        data_dict = {'沪市': 'shclose', '深市': 'szclose', '策略': 'capital'}
        pictitle = fn_tag
        picfn = output_fn_prefix + fn_tag + '_valid_' + validperiod + '-' + '.png'
        draw_equity_curve(pd_buy, data_dict, picfn, pictitle, pic_size=[18, 9], dpi=72, font_size=15)

        ar = annual_return(pd_buy['date'], pd_buy['capital'])
        print('valid days:', validcnt, validperiod, '年化收益率:', ar)
        md = max_drawdown(pd_buy['date'], pd_buy['capital'])
        print('valid days:', validcnt, validperiod, '最大回撤为:', md, '\n')
        return rf_best_threshold, df_topN
    else:
        print("\n ", fn_tag, "Model no good, bypass backtest")
        if '_stk_' in fn_tag:
            return 9999, new_pd
        else:
            return 9999, pd.DataFrame()

def exploreBestThreshold(df,fn_tag, topn):

    rfcls = df['RFclass'].unique()
    rfcls.sort()
    allrows = len(df)
    alldays = len(df['date'].unique())
    exploreResult = pd.DataFrame()
    for rf in rfcls:
        if rf>0:
            temp = df[df['RFclass']>=rf]
            selected = len(temp)/allrows*100
            meanpct = temp['aft1D'].mean()
            temp = temp.pivot_table(index=['date'], aggfunc={'sname': len, 'aft1D': 'mean'}).reset_index()
            temp.columns = ['date','aft1D','cnt']
            temp['buycnt'] = temp.apply(lambda x: topn if x.cnt>topn else x.cnt, axis=1)
            temp['buygain'] = temp.apply(lambda x: x.buycnt*x.aft1D, axis=1)
            if 'aft10' in fn_tag.lower():
                temp['capitalpct'] = temp.apply(lambda x: x.buygain/1000/topn-stkrate/10/topn*x.buycnt, axis=1)
            elif 'aft5d' in fn_tag.lower():
                temp['capitalpct'] = temp.apply(lambda x: x.buygain/500/topn-stkrate/5/topn*x.buycnt, axis=1)
            elif 'aft1' in  fn_tag.lower():
                temp['capitalpct'] = temp.apply(lambda x: x.buygain/100/topn-stkrate/topn*x.buycnt,axis=1)
            else:
                print('unexpected fn_tag', fn_tag)

            temp['capital'] = (1+temp['capitalpct']).cumprod()
            efficiency = list(temp['capital'])[-1]/temp['buycnt'].sum()
            exploreResult = pd.concat([exploreResult, pd.DataFrame({'rf':rf,'selected':[selected],
                'meanpct': [meanpct], 'alldays':[alldays],  'days': [round(len(temp)/alldays*100,0)],
                'buycnt':[temp['buycnt'].sum()], 'finalgain': [(list(temp['capital'])[-1]-1)*100],
                'perBuy':[efficiency]
            })])
    # exploreResult.to_csv('outputQ\\exploreResult_'+fn_tag+'.csv')
    print("maxselect:", topn)
    print(exploreResult[exploreResult['rf']>=0])
    # exploreResult.to_csv('outputQ\\exploreResult_'+fn_tag+'.csv',encoding='gbk',index=False)

    if 'aft10' in fn_tag.lower():
        exploreResult = exploreResult[(exploreResult['finalgain']>=6.) & (exploreResult['selected']<5) & (exploreResult['meanpct']>=2.)]
    elif 'aft5' in fn_tag.lower():
        exploreResult = exploreResult[(exploreResult['finalgain']>=6.) & (exploreResult['selected']<5) & (exploreResult['meanpct']>=1.)]
    elif 'aft1' in fn_tag.lower() and '_stk_' in fn_tag.lower():
        exploreResult = exploreResult[(exploreResult['finalgain']>=2.0) & (exploreResult['selected']<5)]
    elif 'aft1' in fn_tag.lower():
        exploreResult = exploreResult[(exploreResult['finalgain']>=2.0) & (exploreResult['selected']<5) & (exploreResult['meanpct']>=0.4)]
    else:
        print('unexpected fn_tag', fn_tag)

    # exploreResult = exploreResult[(exploreResult['finalgain']>0) & (exploreResult['selected']<10)]
    if len(exploreResult)>0:
        bestrow = exploreResult[exploreResult['finalgain'] == exploreResult['finalgain'].max()]
        rfthreshold = list(bestrow['rf'])[0]
    else:
        print(fn_tag, 'model no good,  set rfthreshold to ', rfcls[-1]+1)
        return rfcls[-1]+1

    print(fn_tag, 'rfthreshold', rfthreshold)

    return rfthreshold

def plotTestPctchg2Aft1d(fn, df, ylimit=20):

    df_selected = df[(df['RF_flag'] == 1)]
    df_others = df[df['RF_flag'] == 0]

    plt.figure(1, figsize=(15, 10))
    plt.scatter(df_others["pctChg"], df_others["aft1D"], s=1, color='g', alpha=0.2, label='other')
    plt.scatter(df_selected["pctChg"], df_selected["aft1D"], s=4, color='r', alpha=1, label='RFpos')

    plt.legend()
    plt.grid()
    plt.title(fn)
    plt.ylim((-ylimit, ylimit))
    plt.xlabel('pctChg')
    plt.ylabel('aft1D')
    plt.savefig(fn)
    plt.cla()
    plt.clf()
    plt.close()
    # plt.show()

def plotTestPred2Aft1d(fn, df, ylimit=20):

    plt.figure(1, figsize=(15, 10))
    # plt.scatter(df["forest"], df["aft1D"], s=4, color='r', alpha=1, label='RFpos')
    plt.scatter(df["forest"], df["aft1D"], s=1, color='g', alpha=0.2, label='predicted')
    # sns.lmplot(x='forest', y='aft1D', data=df, height=10, aspect=1.5, scatter_kws={"s":4,"alpha":0.5}, order=3)
    plt.legend()
    plt.grid()
    plt.ylim((-ylimit, ylimit))
    plt.xlabel('forest')
    plt.ylabel('aft1D')
    plt.title(fn)
    plt.savefig(fn)
    plt.cla()
    plt.clf()
    plt.close()
    # plt.show()

def draw_equity_curve(df,data_dict,fn, title='', pic_size=[18,9],dpi=72,font_size=15):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(pic_size[0],pic_size[1]),dpi=dpi)
    # plt.xticks(fontsize=font_size)
    # plt.xticks(fontsize=font_size)
    for key in data_dict:
        # plt.plot(df[time],df[data_dict[key]],label=key)
        plt.plot(df['date'],df[data_dict[key]],label=key)
    plt.legend(fontsize=font_size)
    plt.grid(ls='--',which='both',axis='both')
    plt.tick_params(axis='x',labelsize=15)
    plt.title(fn+'  '+title)
    plt.savefig(fn)
    # plt.show()
    plt.cla()
    plt.clf()

def performance(ret, benchmark, rf=0.04):
    import empyrical
    max_drawdown = empyrical.max_drawdown(ret)
    total_return = empyrical.cum_returns_final(ret)
    annual_return = empyrical.annual_return(ret)
    sharpe_ratio = empyrical.sharpe_ratio(ret, risk_free=(1+rf**(1/252)-1))
    alpha,beta = empyrical.alpha_beta(ret,benchmark)
    return {'total_return': total_return, 'annual_return': annual_return,
        'max_drawdown':max_drawdown, 'sharpe-ratio': sharpe_ratio,
        'alpha':alpha, 'beta':beta}

# 计算年化收益率函数
def annual_return(date_line, capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :return: 输出在回测期间的年化收益率
    """
    # 将数据序列合并成dataframe并按日期排序
    df = pd.DataFrame({'date': date_line, 'capital': capital_line})
    days = (df['date'].iloc[-1]-df['date'].iloc[0]).days
    # 计算年化收益率
    # annual = (df['capital'].iloc[-1] / df['capital'].iloc[0]) ** (360/days) - 1
    annual = (df['capital'].iloc[-1]) ** (360/days) - 1
    # print('年化收益率', annual)
    return round(annual,3)

# 计算最大回撤函数
def max_drawdown(date_line, capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :return: 输出最大回撤及开始日期和结束日期
    """
    # 将数据序列合并为一个dataframe并按日期排序
    df = pd.DataFrame({'date': date_line, 'capital': capital_line})

    df['max2here'] = df['capital'].cummax()  # 计算当日之前的账户最大价值
    df['dd2here'] = df['capital'] / df['max2here'] - 1  # 计算当日的回撤
    #  计算最大回撤和结束时间
    temp = df.sort_values(by='dd2here').iloc[0][['date', 'dd2here']]
    max_dd = temp['dd2here']
    end_date = temp['date'].strftime('%Y-%m-%d')
    # 计算开始时间
    df = df[df['date'] <= end_date]
    start_date = df.sort_values(by='capital', ascending=False).iloc[0]['date'].strftime('%Y-%m-%d')
    # print('最大回撤为：%f, 开始日期：%s, 结束日期：%s' % (max_dd, start_date, end_date))
    return round(max_dd,4)

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)

def processBatches(seq, batch, data_pull_s, data_pull_e):
    print("processing batch ", seq)
    # codelist = batch.keys()

    batch_start = time.time()
    df_batch = pd.DataFrame()
    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}

    for key, value in batch.items():
        if key[0].lower() == 's':
            continue
        else:
            BaoDataFile = bao_data_path +'BAO-' + key +'.csv'
            if os.path.exists(BaoDataFile):
                df_code = pd.read_csv(BaoDataFile, dtype=dtype_dic, encoding='gbk')
                df_single = df_code[(df_code['date'] >= data_pull_s) & (df_code['date'] <= data_pull_e)]
                df_single['change'] = df_single['pctChg'] / 100
                df_single[['open', 'close', 'high', 'low']] = cal_right_price(df_single, type='后复权')

                df_single = prepQdata(key, value, df_single)
                df_batch = pd.concat([df_batch, df_single])
            else:
                continue

    # df_batch.fillna({'aft1D': 0, 'aft2D': 0, 'aft3D': 0}, inplace=True)
    print('\nbatch:', seq, 'rows:', len(df_batch), ' Data Prepared in ', round(time.time() - batch_start, 0), ' secs.')

    return df_batch

# def prepSingleBK(code, name, df_status,rank_days=40):
#     global today, stocklist
#
#     # add more features with calculation
#     df_status['code'] = code
#     df_status['sname'] = name
#     if 'volume' not in df_status.columns:
#         df_status['volume'] = df_status['vol']
#
#     df_status['preclose'] = df_status['close'].shift(1)
#     df_status['pctChg'] = (df_status['close'] / df_status['preclose'] - 1) * 100
#     # df_status['liutong'] = df_status['amount']/df_status['turn']*100
#     df_status['amount'] = df_status['amount']/100000000
#
#     # try:
#     #     if list(df_status.date)[-1] == datetime.datetime.now().strftime('%Y-%m-%d'):
#     #         df_status.turn[len(df_status) - 1] = df_status.volume[len(df_status)-1] / df_status.volume[len(df_status)-2] * \
#     #                                      df_status.turn[len(df_status) - 2]
#     #         df_status.pctChg[len(df_status)-1] = (df_status.close[len(df_status)-1] / df_status.close[len(df_status)-2] -1)*100
#     # except:
#     #     print(code, name, 'last row date:', list(df_status.date)[-1], 'is not today')
#         # pass
#
#     df_status['opct'] = (df_status['open'] / df_status['preclose'] - 1) * 100
#     df_status['hpct'] = (df_status['high'] / df_status['preclose'] - 1) * 100
#     df_status['lpct'] = (df_status['low'] / df_status['preclose'] - 1) * 100
#
#     df_status['pctR2'] = (df_status["close"]/df_status["close"].shift(2)-1)*100
#     df_status['pctR3'] = (df_status["close"]/df_status["close"].shift(3)-1)*100
#     df_status['pctR4'] = (df_status["close"]/df_status["close"].shift(4)-1)*100
#     df_status['pctR5'] = (df_status["close"]/df_status["close"].shift(5)-1)*100
#     df_status['pctR6'] = (df_status["close"]/df_status["close"].shift(6)-1)*100
#     df_status['pctR7'] = (df_status["close"]/df_status["close"].shift(7)-1)*100
#     df_status['pctR8'] = (df_status["close"]/df_status["close"].shift(8)-1)*100
#
#     df_status['kline'] = df_status.apply(lambda x: 1 if x.close>x.open else 0, axis=1)
#
#     df_status['cp5'] = (df_status["close"] - df_status["close"].rolling(5,min_periods=1).min()) \
#                        / (df_status["close"].rolling(5).max() - df_status["close"].rolling(5).min())
#     df_status['cp20'] = (df_status["close"] - df_status["close"].rolling(20,min_periods=1).min()) \
#                         / (df_status["close"].rolling(20).max() - df_status["close"].rolling(20).min())
#     df_status['cp60'] = (df_status["close"] - df_status["close"].rolling(60,min_periods=1).min()) \
#                         / (df_status["close"].rolling(60).max() - df_status["close"].rolling(60).min())
#
#     df_status['vt5'] = (df_status["volume"] - df_status["volume"].rolling(5,min_periods=1).min()) \
#                        / (df_status["volume"].rolling(5).max() - df_status["volume"].rolling(5).min())
#     df_status['vt10'] = (df_status["volume"] - df_status["volume"].rolling(10,min_periods=1).min()) \
#                         / (df_status["volume"].rolling(10).max() - df_status["volume"].rolling(10).min())
#     df_status['vt20'] = (df_status["volume"] - df_status["volume"].rolling(20,min_periods=1).min()) \
#                         / (df_status["volume"].rolling(20).max() - df_status["volume"].rolling(20).min())
#     df_status['vt60'] = (df_status["volume"] - df_status["volume"].rolling(60,min_periods=1).min()) \
#                         / (df_status["volume"].rolling(60).max() - df_status["volume"].rolling(60).min())
#     df_status['vratio5'] = df_status["volume"] / df_status["volume"].rolling(5,min_periods=1).mean()
#     df_status['vratio10'] = df_status["volume"] / df_status["volume"].rolling(10,min_periods=1).mean()
#     df_status['vratio20'] = df_status["volume"] / df_status["volume"].rolling(20,min_periods=1).mean()
#     df_status['vratio60'] = df_status["volume"] / df_status["volume"].rolling(60,min_periods=1).mean()
#
#     df_status['aft1D'] = df_status["close"].shift(-1)/df_status["close"]*100-100
#     df_status['aft2D'] = df_status["close"].shift(-2)/df_status["close"]*100-100
#     df_status['aft3D'] = df_status["close"].shift(-3)/df_status["close"]*100-100
#     # df_status['aft4D'] = df_status["pctChg"].rolling(4).sum().shift(-4)
#     df_status['aft5D'] = df_status["close"].shift(-5)/df_status["close"]*100-100
#     df_status['aft8D'] = df_status["close"].shift(-8)/df_status["close"]*100-100
#     df_status['aft10D'] = df_status["close"].shift(-10)/df_status["close"]*100-100
#
#     df_status['rank'] = df_status['volume'].rolling(rank_days + 1).apply(lambda x: pd.Series(x).rank().iloc[-1])
#     df_status['vol_rank'] = 2 * (df_status['rank'] - rank_days - 1) / rank_days + 1
#     df_status['bias'] = computeBias(df_status[['close']], 10)
#     df_status['biasgood'] = (df_status['bias']<0).map({True: 1, False: 0})
#     df_status['biasup'] = df_status['bias'] < df_status['bias'].shift(1)
#
#     df_status['ma5'] = df_status["close"].rolling(5).mean()
#     df_status['ma5up'] = (df_status['ma5']>df_status['ma5'].shift(1))
#     df_status['ma5delta'] = (df_status['ma5']/df_status['ma5'].shift(1)-1)*100
#     df_status.ma5up = df_status.ma5up.map({True: 1, False: 0})
#     df_status['abovema5'] = (df_status["close"]>df_status['ma5']).map({True: 1, False: 0})
#
#     df_status['ma10'] = df_status["close"].rolling(10).mean()
#     df_status['ma10up'] = df_status['ma10']>df_status['ma10'].shift(1)
#     df_status['ma10delta'] = (df_status['ma10'] / df_status['ma10'].shift(1) - 1) * 100
#     df_status.ma10up = df_status.ma10up.map({True: 1, False: 0})
#     df_status['abovema10'] = (df_status["close"] > df_status['ma10']).map({True: 1, False: 0})
#
#     df_status['kline'] = df_status.apply(lambda x: 1 if x.ma10up and x.close>x.ma10 else 0, axis=1)
#
#     df_status['ma20'] = df_status["close"].rolling(20).mean()
#     df_status['ma20up'] = df_status['ma20']>df_status['ma20'].shift(1)
#     df_status.ma20up = df_status.ma20up.map({True: 1, False: 0})
#     df_status['ma20delta'] = (df_status['ma20'] / df_status['ma20'].shift(1) - 1) * 100
#     df_status['abovema20'] = (df_status["close"] > df_status['ma20']).map({True: 1, False: 0})
#
#     df_status['indicator'] = df_status.apply(lambda x: calcIndicator(x.ma5, x.ma10, x.ma20), axis=1)
#
#     df_status['kdj'] = computeKDJv2(df_status.copy())
#     df_status['bwidth'], df_status['bupperdelta'] = computeBollv2(df_status.copy())
#     # df_status['rsi'] = computeRSI(df_status)
#     df_status['rsi'] = ta.RSI(df_status['close'], timeperiod=14)
#     # df_status['obv'] = computeOBV(df_status)
#     df_status['obv'] = ta.OBV(df_status['close'].values, df_status['volume'].values)
#     df_status['obv'] = (df_status['obv']>df_status['obv'].shift(2)).map({True: 1, False: 0})
#
#     df_status['week'] = df_status.apply(lambda x: week_day(x.date), axis=1)
#     # df_status['c60cls'] = df_status.apply(lambda x: mybin5(x.cp60), axis=1)
#     # df_status['v60cls'] = df_status.apply(lambda x: mybin5(x.vt60), axis=1)
#
#     return df_status

# def GetStockHis(data_pull_s, data_pull_e):
#
#     print('\n===== start 4k+ stock data pulling ======')
#     print('data_pull_s:', data_pull_s, 'data_pull_e:', data_pull_e)
#
#     sl = pd.read_csv('stocklist_full.csv', dtype=str, encoding='utf8')
#     batchCnt = 8
#     stockPerBatch = len(sl) // batchCnt + 1
#     stockBatchList = []
#
#     for i in range(batchCnt):
#         sl_temp = sl[i * stockPerBatch: (i + 1) * stockPerBatch]
#         stocklist_temp = {}
#         for index, row in sl_temp.iterrows():
#             stocklist_temp[row['code']] = row['sname']
#         stockBatchList.append(stocklist_temp)
#
#     # df_concat = pd.DataFrame()
#     with concurrent.futures.ProcessPoolExecutor() as executor:  # 多进程
#         results = [executor.submit(processStkBatches, i, batch, data_pull_s, data_pull_e ) \
#                    for i, batch in enumerate(stockBatchList)]
#         df_concat = pd.concat([result.result() for result in results], ignore_index=True)
#
#     df_concat = df_concat[(df_concat["pctChg"] < 9.8) & (df_concat["pctChg"] > -9.8)]
#     # print('df_concat shape: ', df_concat.shape)
#     # print('df_concat.corr()\n', df_concat.corr())
#     df_concat.corr().to_csv('..\\..\\daily_output\\TdxBK2stk_Qtr_stklist_corr_' + datetime.datetime.now().strftime('%Y%m%d') + '.csv', index=False, encoding='gbk')
#
#     return df_concat


# 板块A关联到个股B，求笛卡尔积

def getMergeAB(A,B, colkey):
    newDf = pd.DataFrame()
    for _,A_row in A.iterrows():
        # key = A_row[colkey]
        tmp = B[B[colkey]==A_row[colkey]]
        tmp['date'] = A_row['date']
        newDf = pd.concat([newDf, tmp])
    return newDf



