import json, datetime, os,re
import numpy as np
import warnings
import time, requests
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from utils.mplfinance import *
import configparser

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
from utils.tdx_hosts import hq_hosts, Exhq_hosts
from utils.tdx_indicator import *

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

pd.options.mode.chained_assignment = None
pd.set_option('display.precision', 4)
pd.set_option('display.float_format',  '{:.2f}'.format)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('expand_frame_repr', False)
warnings.filterwarnings('ignore')
np.random.seed(42)


if not os.path.exists('output'):
    os.makedirs('output')


class mytdxData(object):

    def __init__(self):

        api = TdxHq_API(heartbeat=True)
        apilist=pd.DataFrame(hq_hosts)
        apilist.columns=['name','ip','port']
        if self.TestConnection(api, 'HQ', apilist['ip'].values[0], apilist['port'].values[0]) == False:  # or \
            if self.TestConnection(api, 'HQ', apilist['ip'].values[1], apilist['port'].values[1]) == False:  # or \
                if self.TestConnection(api, 'HQ', apilist['ip'].values[2], apilist['port'].values[2]) == False:  # or \
                    if self.TestConnection(api, 'HQ', apilist['ip'].values[3], apilist['port'].values[3]) == False:  # or \
                        print('All HQ server Failed!!!')
                    else:
                        print(f'connection to HQ server[3]!{apilist["name"].values[3]}')
                else:
                    print(f'connection to HQ server[2]!{apilist["name"].values[2]}')
            else:
                print(f'connection to HQ server[1]!{apilist["name"].values[1]}')
        else:
            print(f'connection to HQ server[0]!{apilist["name"].values[0]}')

        Exapi = TdxExHq_API(heartbeat=True)
        exapilist=pd.DataFrame(Exhq_hosts)
        exapilist.columns=['name','ip','port']
        if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[0], exapilist['port'].values[0]) == False:  # or \
            if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[1], exapilist['port'].values[1]) == False:  # or \
                if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[2], exapilist['port'].values[2]) == False:  # or \
                    if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[3], exapilist['port'].values[3]) == False:  # or \
                        print('All ExHQ server Failed!!!')
                    else:
                        print(f'connection to ExHQ server[3]!{exapilist["name"].values[3]}')
                else:
                    print(f'connection to ExHQ server[2]!{exapilist["name"].values[2]}')
            else:
                print(f'connection to ExHQ server[1]!{exapilist["name"].values[1]}')
        else:
            print(f'connection to ExHQ server[0]!{exapilist["name"].values[0]}')

        self.api = api
        self.Exapi = Exapi
        self.useless_cols = ['year','month','day','hour','minute','preclose','change']
        self.period_dict =  {"5min": 0, "15min": 1, "30min": 2, "60min": 3, "week": 5,
                  "month": 6, "1min": 8, "day": 9, "quater": 10, "year": 11}

    def TestConnection(self, Api, type, ip, port):
        if type == 'HQ':
            try:
                is_connect = Api.connect(ip, port)
            except Exception as e:
                print('connect to HQ Exception!')
                exit(0)
            return False if is_connect is False else True

        elif type == 'ExHQ':
            try:
                is_connect = Api.connect(ip, port)
            except Exception as e:
                print('connect to Ext HQ Exception!')
                exit(0)
            return False if is_connect is False else True

    def cal_right_price(self, input_stock_data, type='前复权'):
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

    def get_xdxr_EM(self,code):
        if len(code)!=5:
            return pd.DataFrame()
        url = 'https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=RPT_HKF10_MAIN_DIVBASIC'+ \
              '&columns=SECURITY_CODE,UPDATE_DATE,REPORT_TYPE,EX_DIVIDEND_DATE,DIVIDEND_DATE,TRANSFER_END_DATE,YEAR,PLAN_EXPLAIN,IS_BFP'+ \
              '&quoteColumns=&filter=(SECURITY_CODE="'+code+'")(IS_BFP="0")&pageNumber=1&pageSize=3&sortTypes=-1,-1'+ \
              '&sortColumns=NOTICE_DATE,EX_DIVIDEND_DATE&source=F10&client=PC&v=043409724372028'
        try:
            res = requests.get(url)
            data1 = pd.DataFrame(json.loads(res.text)['result']['data'])
            if len(data1)==0:
                return pd.DataFrame()
            else:
                data1.rename(columns={'EX_DIVIDEND_DATE':'date','SECURITY_CODE':'code','REPORT_TYPE':'type','PLAN_EXPLAIN':'deal'}, inplace=True)
                data1 = data1[data1['type'].str.contains('分配')]
                data1['date'] = data1['date'].apply(lambda x: x.replace('/','-'))
                data1['deal'] = data1['deal'].apply(lambda x: float(re.findall(r'\d+\.\d+(?=[^.\d]*$)',x)[-1]))
                return data1
        except:
            return pd.DataFrame()

    def get_market_code(self,code):

        mkt = None
        fuquan = False
        isIndex = False

        if '#' in code:
            mkt = int(code.split('#')[0])
            code = code.split('#')[1]

        if code[:2] == 'ZS':
            code = code[-6:]
            isIndex = True
            fuquan = False

        if code.isdigit() and len(code)==6: # A股
            if isIndex==True:
                mkt = 1 if code[:2] == '00' else 0 # 上证指数，深证成指
            elif code[:2] in ['15','00','30','16','12','39','18']: # 深市
                mkt = 0 # 深交所
                fuquan = True
            elif code[:2] in ['51','58','56','60','68','50','88','11','99']:
                mkt = 1 # 上交所
                fuquan = True
            elif code[:2] in ['43','83','87']:
                mkt = 2 # 北交所pass
                fuquan = True
            else:
                pass

            if code[:2] in ['39','88','99']:
                isIndex = True
                fuquan = False

        elif code.isdigit() and len(code)==5: # 港股
            mkt = 71
            fuquan = True
        elif code.isdigit() and len(code)==8: # 期权
            if code[:2] == '10':
                mkt = 8
            elif code[:2] == '90':
                mkt = 9
            else:
                mkt = None
        elif code.isalpha() : # 美股
            mkt = None
        elif len(code) == 5 and code[0]=='U':    # 期权指标如持仓比
            mkt = 68
        else:
            mkt = None
        return mkt,code,fuquan,isIndex


    def fuquan202409(self, code, backset, qty=200, period=9):

        mkt,code,fuquan,isIndex = self.get_market_code(code)

        if period!=9:           # 如果不是日线级别，跳过复权直接返回。
            fuquan = False

        if code[:2] in ['88','11','12','39','99','zz','zs']:  # 指数\债券不复权
            fuquan = False

        if mkt is None:
            print(code, 'unknown code')
            return pd.DataFrame()

        if mkt in [0,1,2]:  # A股
            if qty<=600:
                if isIndex==False:
                    df_k = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 0 + backset, qty))
                else:
                    df_k = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 0 + backset, qty))
            elif isIndex==False:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp, df_k])
                temp = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            else:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp, df_k])
                temp = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])

            if len(df_k) ==0:
                return pd.DataFrame()

            if fuquan:   # A股复权
                df_fuquan = self.api.get_xdxr_info(mkt, code)
                if df_fuquan is None:
                    return df_k
                elif len(df_fuquan) == 0:
                    return df_k
                else:
                    df_fuquan = pd.DataFrame(df_fuquan, index=[i for i in range(len(df_fuquan))])
                    df_fuquan['date'] = df_fuquan.apply(
                        lambda x: str(x.year) + '-' + str(x.month).zfill(2) + '-' + str(x.day).zfill(2), axis=1)

                    df_k['preclose'] = df_k['close'].shift(1)
                    df_k['change'] = df_k['close']/df_k['preclose']-1
                    if 'date' not in df_k.columns:
                        df_k['date'] = df_k['datetime'].apply(lambda x: x[:10])
                    for i, row in df_fuquan.iterrows():
                        if row['date'] not in list(df_k['date']):
                            continue
                        elif row['name'] == '除权除息':
                            preclose = df_k.loc[df_k['date'] == row['date'], 'preclose'].values[0]
                            thisclose = df_k.loc[df_k['date'] == row['date'], 'close'].values[0]
                            if row['fenhong'] > 0 and row['songzhuangu'] > 0:
                                change_new = (thisclose * (row['songzhuangu'] / 10 + 1) + row[
                                    'fenhong'] / 10 ) / preclose - 1
                                df_k.loc[df_k['date'] == row['date'], 'change'] = change_new
                            elif row['fenhong'] > 0:
                                change_new = (thisclose + row['fenhong'] / 10) / preclose - 1
                                df_k.loc[df_k['date'] == row['date'], 'change'] = change_new
                            elif row['songzhuangu'] > 0:
                                change_new = (thisclose * (row['songzhuangu'] / 10 + 1)) / preclose - 1
                                df_k.loc[df_k['date'] == row['date'], 'change'] = change_new
                        elif row['name'] == '扩缩股':
                            preclose = df_k.loc[df_k['date'] == row['date'], 'preclose']
                            thisclose = df_k.loc[df_k['date'] == row['date'], 'close']
                            change_new = (thisclose * row['suogu']) / preclose - 1
                            df_k.loc[df_k['date'] == row['date'], 'change'] = change_new
                        elif row['name'] in ['股本变化', '非流通股上市', '转配股上市', '送配股上市']:
                            continue
                        else:
                            print(code, 'unknown name:', row['name'])
                    df_k[['open', 'close', 'high', 'low']] = self.cal_right_price(df_k, type='前复权')
                    return df_k
            else: # A股不复权
                return df_k

        elif mkt in [8,9]:  # 期权
            if qty<=600:
                df_k = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 0 + backset, qty))
            else:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp, df_k])
                temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            return df_k

        elif mkt == 71:  # 港股
            if qty<=600:
                df_k = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 0 + backset, qty))
            else:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp,df_k ])
                temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])

            if fuquan : # 港股通复权
                df_fuquan = self.get_xdxr_EM(code)  # 港股通复权
                if len(df_fuquan)==0:
                    return df_k
                for i,row in df_fuquan.iterrows():
                    if row['date'] not in list(df_k['date']):
                        continue
                    preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
                    thisclose = df_k.loc[df_k['date']==row['date'], 'close']
                    change_new = (thisclose+row['deal']*0.99)/preclose-1
                    df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                df_k[['open', 'close', 'high', 'low']] = self.cal_right_price(df_k, type='前复权')
                return df_k
            else: # 港股通不复权
                return df_k

        elif mkt == 68:  # 期权持仓比
            if qty<=600:
                df_k = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 0 + backset, qty))
            else:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp,df_k ])
                temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            return df_k


        else:   # 未知 mkt
            print(code, f'unknown {mkt}')
            return pd.DataFrame()

    def get_disk_port_data_stock(self,stock):
        mkt,code,fuquan,isIndex = self.get_market_code(stock)

        if mkt in [0,1,2]:
            result =pd.DataFrame(self.api.get_security_quotes([(mkt, stock)]))
            result.rename(columns={'code':'code','last_close':'昨收','bid1':'买一','ask1':'卖一',
                                   'bid2': '买二', 'ask2': '卖二', 'bid3': '买三', 'ask3': '卖三',
                                   'bid4': '买四', 'ask4': '卖四', 'bid5': '买五', 'ask5': '卖五',
                                   'bid_vol1': '买一量', 'ask_vol1': '卖一量',
                                   'bid_vol2': '买二量', 'ask_vol2': '卖二量', 'bid_vol3': '买三量',
                                   'ask_vol3': '卖三量', 'bid_vol4': '买四量', 'ask_vol4': '卖四量', 'bid_vol5': '买五量',
                                   'ask_vol5': '卖五量'}, inplace=True)
        else:
            result = pd.DataFrame(self.Exapi.get_instrument_quote(mkt,stock))
            result.rename(columns={'code':'code','pre_close':'昨收','bid1':'买一','ask1':'卖一',
                                   'bid2': '买二', 'ask2': '卖二', 'bid3': '买三', 'ask3': '卖三',
                                   'bid4': '买四', 'ask4': '卖四', 'bid5': '买五', 'ask5': '卖五',
                                   'bid_vol1': '买一量', 'ask_vol1': '卖一量',
                                   'bid_vol2': '买二量', 'ask_vol2': '卖二量', 'bid_vol3': '买三量',
                                   'ask_vol3': '卖三量', 'bid_vol4': '买四量', 'ask_vol4': '卖四量', 'bid_vol5': '买五量',
                                   'ask_vol5': '卖五量'}, inplace=True)
        if len(result)==0:
            return pd.DataFrame()
        return result[['code','昨收','卖五','卖四','卖三','卖二','卖一','买一','买二','买三','买四','买五',
                       '卖五量','卖四量','卖三量','卖二量','卖一量','买一量','买二量','买三量','买四量','买五量']].T


    def get_minute_data(self, code, day):
        mkt,code,fuquan,isIndex = self.get_market_code(code)

        if isinstance(day, str):
            day = int(day.replace('-',''))

        if mkt in [0,1,2] and isIndex==False:
            data = pd.DataFrame(self.api.get_history_minute_time_data(mkt, code, day))
        elif mkt in [0,1,2] and isIndex==True:
            data = pd.DataFrame(self.api.get_history_minute_time_data(mkt, code, day))
        else:
            data = pd.DataFrame()
        return data

    def get_kline_data(self,code, backset=0, klines=200, period=9):
        df=self.fuquan202409(code, backset, klines, period)

        if len(df)==0:
            return pd.DataFrame()

        if '成交额' not in df.columns:
            df.rename(columns={'vol': 'volume', 'amount': '成交额'}, inplace=True)
            df['preclose'] = df['close'].shift(1)
            df['振幅'] = df.apply(lambda x: (x['high'] - x['low']) / x['preclose'], axis=1)
            df['涨跌幅'] = df.apply(lambda x: x['close'] / x['preclose'] - 1, axis=1)
        df.dropna(subset=['preclose'],inplace=True)
        for col in self.useless_cols:
            if col in df.columns:
                del df[col]
        df.reset_index(drop=True, inplace=True)
        return df

class analyzer(object):

    def __init__(self,df):
        self.data = df

    def six_pulse_excalibur(self):
        df = self.data
        HIGH = df['high'].values
        LOW = df['low'].values
        CLOSE = df['close'].values

        DIFF = EMA(CLOSE, 8) - EMA(CLOSE, 13)
        DEA = EMA(DIFF, 5)
        ABC1 = DIFF > DEA

        RSV1 = (CLOSE - LLV(LOW, 8)) / (HHV(HIGH, 8) - LLV(LOW, 8)) * 100
        K = SMA(RSV1, 3, 1)
        D = SMA(K, 3, 1)
        ABC2=K>D

        LC = REF(CLOSE, 1)
        RSI1 = (SMA(MAX(CLOSE - LC, 0), 5, 1)) / (SMA(ABS(CLOSE - LC), 5, 1)) * 100
        RSI2 = (SMA(MAX(CLOSE - LC, 0), 13, 1)) / (SMA(ABS(CLOSE - LC), 13, 1)) * 100
        ABC3=RSI1>RSI2

        RSV = -(HHV(HIGH, 13) - CLOSE) / (HHV(HIGH, 13) - LLV(LOW, 13)) * 100
        LWR1 = SMA(RSV, 3, 1)
        LWR2 = SMA(LWR1, 3, 1)
        ABC4=LWR1>LWR2

        BBI = (MA(CLOSE, 3) + MA(CLOSE, 6) + MA(CLOSE, 12) + MA(CLOSE, 24)) / 4
        ABC5=CLOSE>BBI

        MTM = CLOSE - REF(CLOSE, 1)
        MMS = 100 * EMA(EMA(MTM, 5), 3) / EMA(EMA(ABS(MTM), 5), 3)
        MMM = 100 * EMA(EMA(MTM, 13), 8) / EMA(EMA(ABS(MTM), 13), 8)
        ABC6=MMS>MMM

        df['condmacd'] = [1 if i == True else 0 for i in ABC1]
        df['conddkj'] = [1 if i == True else 0 for i in ABC2]
        df['condrsi'] = [1 if i == True else 0 for i in ABC3]
        df['condbbi'] = [1 if i == True else 0 for i in ABC4]
        df['condlwr'] = [1 if i == True else 0 for i in ABC5]
        df['condmtm'] = [1 if i == True else 0 for i in ABC6]

        df1 = df[['condmacd', 'conddkj', 'condrsi', 'condbbi', 'condlwr', 'condmtm']]  # ,'mean_30_mean_60']]
        # df['score'] = df1.sum(axis=1).tolist()

        return df1.sum(axis=1).tolist()

    def get_kline_data(self,code, backset=0, klines=200, period=9):
        df=self.fuquan202409(code, backset, klines, period)

        if len(df)==0:
            return pd.DataFrame()

        if '成交额' not in df.columns:
            df.rename(columns={'vol': 'volume', 'amount': '成交额'}, inplace=True)
            df['preclose'] = df['close'].shift(1)
            df['振幅'] = df.apply(lambda x: (x['high'] - x['low']) / x['preclose'], axis=1)
            df['涨跌幅'] = df.apply(lambda x: x['close'] / x['preclose'] - 1, axis=1)
        df.dropna(subset=['preclose'],inplace=True)
        for col in self.useless_cols:
            if col in df.columns:
                del df[col]
        df.reset_index(drop=True, inplace=True)
        return df

def getOptionsTformat(df_4T):

    field_map3 = {'f14':'Cname','f12':'Ccode','f2':'Cprice', 'f3':'CpctChg','f4':'C涨跌额','f108':'C持仓量','f5':'Cvol','f249':'Civ','f250':'C折溢价率','f161':'行权价',
                  'f340':'Pname','f339':'Pcode','f341':'Pprice','f343':'PpctChg','f342':'P涨跌额','f345':'P持仓量','f344':'Pvol','f346':'Piv','f347':'P折溢价率'}

    df_T_data = pd.DataFrame()
    for etfcode, expiredate in zip(df_4T['ETFcode'],df_4T['到期日']):
        code= '1.'+etfcode if etfcode[0]=='5' else '0.'+etfcode

        url3 = 'https://push2.eastmoney.com/api/qt/slist/get?cb=jQuery112400098284603835751_1695513185234&'+ \
               'secid='+code+'&exti='+expiredate[:6]+ \
               '&spt=9&fltt=2&invt=2&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fields=f1,f2,f3,f4,f5,f12,f13,f14,f108,f152,f161,'+ \
               'f249,f250,f330,f334,f339,f340,f341,f342,f343,f344,f345,f346,f347&fid=f161&pn=1&pz=20&po=0&wbp2u=|0|0|0|web&_=1695513185258'
        res = requests.get(url3)
        tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
        single = pd.DataFrame(json.loads(tmp)['data']['diff'])
        df_T_data = pd.concat([df_T_data,single])
    df_T_data.rename(columns = field_map3,inplace=True)
    df_T_data = df_T_data[list(field_map3.values())]

    return df_T_data

def getOptionsRiskData():

    field_map4 =  {"f2": "最新价","f3": "涨跌幅","f12": "code","f14": "name", "f301": "到期日",
                   "f302": "杠杆比率","f303": "实际杠杆","f325": "Delta","f326": "Gamma","f327": "Vega","f328": "Theta","f329": "Rho"}

    df_risk = pd.DataFrame()
    for i in range(1,11,1):
        url4 = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112308418460865815227_1695516975860&fid=f3&po=1&'+ \
               'pz='+'50'+'&pn='+str(i)+'&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5'+ \
               '&fields=f1,f2,f3,f12,f13,f14,f302,f303,f325,f326,f327,f329,f328,f301,f152,f154&fs=m:10'
        res = requests.get(url4)
        tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
        if len(tmp)<100:
            continue
        single = pd.DataFrame(json.loads(tmp)['data']['diff'])
        df_risk = pd.concat([df_risk,single])

    df_risk.rename(columns = field_map4,inplace=True)
    df_risk = df_risk[list(field_map4.values())]

    return df_risk

def getAllOptionsV3():

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0',
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Cookie": "qgqp_b_id=435b18200eebe2cbb5bdd3b3af2db1b1; intellpositionL=522px; intellpositionT=1399.22px; em_hq_fls=js; pgv_pvi=6852699136; st_pvi=73734391542044; st_sp=2020-07-27%2010%3A10%3A43; st_inirUrl=http%3A%2F%2Fdata.eastmoney.com%2Fhsgt%2Findex.html",
        "Host": "push2.eastmoney.com",
    }

    field_map0 = {'f12':'code', 'f14':'name','f301':'到期日','f331':'ETFcode', 'f333':'ETFname',
                  'f2':'close','f3':'pctChg','f334':'ETFprice','f335':'ETFpct','f337':'平衡价',
                  'f250':'溢价率','f161':'行权价'}#,'f47':'vol','f48':'amount','f133':'allvol'}

    url1 = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112307429657982724098_1687701611430&fid=f250'+ \
           '&po=1&pz=1000&pn=1&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5'+ \
           '&fields=f1,f2,f3,f12,f13,f14,f161,f250,f330,f331,f332,f333,f334,f335,f337,f301,f152&fs=m:10'
    res = requests.get(url1, headers=header)
    tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
    data1 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    data1.rename(columns = field_map0,inplace=True)

    url2 = url1[:-1] + '2'
    res = requests.get(url2,headers=header)
    tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
    data2 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    data2.rename(columns = field_map0,inplace=True)

    data = pd.concat([data1, data2])
    data = data[list(field_map0.values())]

    data['market'] = data['ETFcode'].apply(lambda x: '沪市' if x[0]=='5' else '深市')
    data['direction'] = data['name'].apply(lambda x: 'call' if '购' in x else 'put')
    data['due_date'] = data['到期日'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d').date())
    data['dte'] = data['due_date'].apply(lambda x: (x-datetime.datetime.now().date()).days)
    data['close'] = data['close'].astype(float)
    data['到期日'] = data['到期日'].astype(str)
    data['行权pct'] = data.apply(lambda x:round(x['行权价']/x['ETFprice']*100-100,2),axis=1)

    df_4T = data.pivot_table(index=['ETFcode','到期日'],values=['name'],aggfunc=['count']).reset_index()
    df_4T.columns = ['ETFcode','到期日','数量']


    df_T_format = getOptionsTformat(df_4T)
    tempc = df_T_format[['Ccode','C持仓量','Cvol']]
    tempc.columns = ['code','持仓量','vol']
    tempp = df_T_format[['Pcode','P持仓量','Pvol']]
    tempp.columns = ['code','持仓量','vol']
    temp = pd.concat([tempc, tempp])
    temp['vol'] = temp['vol'].astype(int)
    data = pd.merge(data, temp, on='code',how='left')
    data['amount'] = data['close']*data['vol']

    df_risk = getOptionsRiskData()
    data = pd.merge(data, df_risk[['code','杠杆比率','实际杠杆', 'Delta','Gamma','Vega','Theta','Rho']], on='code',how='left')

    return data

def getMyOptions():
    global dte_high, dte_low,close_Threshold, opt_fn, tdxdata

    # now = pd.DataFrame(api.get_index_bars(8, 1, '999999', 0, 20))
    now = tdxdata.get_kline_data('999999',0,20,8)

    current_datetime = datetime.datetime.strptime(now['datetime'].values[-1],'%Y-%m-%d %H:%M')

    earlymorning = True if (time.strftime("%H%M", time.localtime()) >= '0930' and time.strftime("%H%M", time.localtime()) <= '0935') else False

    if earlymorning and os.path.exists(opt_fn_last):
        data = pd.read_csv(opt_fn_last, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    elif os.path.exists(opt_fn):
        modified_timestamp = os.path.getmtime(opt_fn)
        modified_datetime = datetime.datetime.fromtimestamp(modified_timestamp)
        time_delta = current_datetime - modified_datetime
        gap_seconds = time_delta.days*24*3600 + time_delta.seconds
        if gap_seconds < 1000:
            print('reusing option file')
            data = pd.read_csv(opt_fn, encoding='gbk',dtype={'ETFcode':str,'code':str})
        else:
            try:
                data = getAllOptionsV3()
                print('New option file')
                data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
            except:
                print('update failed, reusing option file')
                data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    else:
        print('New option file ' + opt_fn)
        data = getAllOptionsV3()
        data.to_csv(opt_fn,encoding='gbk',index=False, float_format='%.4f')

    data.fillna(0,inplace=True)
    amtlist = data['amount'].values.tolist()
    amtlist.sort()
    amtthreshold = amtlist[-200]

    data = data[data['amount']>amtthreshold]
    data.sort_values(by='amount',ascending=False,inplace=True)
    data['itm'] = data.apply(lambda x: max(0,x.ETFprice-x['行权价']) if x.direction=='call' else max(0,x['行权价']-x.ETFprice),axis=1)
    data['otm'] = data.apply(lambda x: x.close-x.itm,axis=1)

    png_dict = {}
    for key in etf_dict.keys():
        etfcode = etfcode_dict[key]
        call = data[(data['ETFcode']==etfcode) & (data['direction']=='call') & (data['dte']>dte_low) & (data['dte']<dte_high) & (data['close']>close_Threshold)][:1]
        put = data[(data['ETFcode']==etfcode) & (data['direction']=='put') & (data['dte']>dte_low) & (data['dte']<dte_high) & (data['close']>close_Threshold)][:1]
        if len(call) == 0:
            tmpstr = '认购:流动性过滤为空   '
        else:
            tmpstr = '认购:' + call['code'].values[0] + '_' + call['name'].values[0] + '_' + str(
                call['close'].values[0]) + ' =itm' + str(int(call['itm'].values[0]*10000)) + '+' + str(int(call['otm'].values[0]*10000)) + \
                ' 杠杆:'+str(int(call['实际杠杆'].values[0]))
        if len(put) == 0:
            tmpstr += '\n认沽:流动性过滤为空'
        else:
            tmpstr += '\n认沽:' + put['code'].values[0] + '_' + put['name'].values[0] + '_' + str(
                put['close'].values[0]) + ' =itm' + str(int(put['itm'].values[0]*10000)) + '+' + str(int(put['otm'].values[0]*10000)) + \
                ' 杠杆:'+str(int(put['实际杠杆'].values[0]))

        png_dict[key] = tmpstr


    return png_dict,data

def MINgetDPindex():
    global factor, lastday_amount

    # day_sh =  pd.DataFrame(api.get_index_bars(9, 1, '999999', 0, 30))
    day_sh =  tdxdata.get_kline_data('999999',0,30,9)
    lastday_amount_sh =  day_sh['成交额'].values[-2]
    lastday_amount_sz =  tdxdata.get_kline_data('399001',0,30,9)['成交额'].values[-2]
    lastday_amount = (lastday_amount_sh + lastday_amount_sz)/100000000

    if 'amount' not in day_sh.columns:
        day_sh['amount'] = day_sh['成交额']
    if 'vol' not in day_sh.columns:
        day_sh['vol'] = day_sh['volume']
    datelast = day_sh['datetime'].values[-2]
    preclose = day_sh[day_sh['datetime']==datelast]['close'].values[-1]

    # df_sh =  pd.DataFrame(api.get_index_bars(8, 1, '999999', 0, 300))
    df_sh =  tdxdata.get_kline_data('999999',0,300,8)
    if 'amount' not in df_sh.columns:
        df_sh['amount'] = df_sh['成交额']
    if 'vol' not in df_sh.columns:
        df_sh['vol'] = df_sh['volume']

    df_sh['date'] = df_sh['datetime'].apply(lambda x: x[:10])
    df_sh['time'] = df_sh['datetime'].apply(lambda x: x[11:])
    df_sh = df_sh[(df_sh['datetime']>datelast)]
    df_sh.reset_index(drop=True,inplace=True)

    # df_sz =  pd.DataFrame(api.get_index_bars(8, 0, '399001', 0, 300))
    df_sz =  tdxdata.get_kline_data('399001',0,300,8)
    if 'amount' not in df_sz.columns:
        df_sz['amount'] = df_sz['成交额']
    df_sz['date'] = df_sz['datetime'].apply(lambda x: x[:10])
    df_sz['time'] = df_sz['datetime'].apply(lambda x: x[11:])
    df_sz = df_sz[(df_sz['datetime']>datelast)]
    df_sz.reset_index(drop=True,inplace=True)
    df_sz.rename(columns={'amount':'amountsz'}, inplace=True)

    data = pd.merge(df_sh[['datetime','amount','vol','close','high','low']], df_sz[['datetime','amountsz']], how='inner')
    data['allamt'] = data['amount']+data['amountsz']
    data['amt'] = data['close']*data['vol']
    data['amtcum'] = data['amt'].cumsum()
    data['volcum'] = data['vol'].cumsum()
    data['avg'] = data['amtcum']/data['volcum']
    data['amt'] = data['amount'] + data['amountsz']
    data['amtcum'] = data['amt'].cumsum()
    data['factor'] = factor[:len(data)]
    data['amttrend'] = data['factor']*data['amtcum']
    # data['amttrend'] = data['amttrend'].ffill()
    # data.iloc[-1, data.columns.get_loc('amttrend')] = np.nan

    data.drop(columns=['amtcum','volcum','amt','factor'], inplace=True)

    return data, preclose

def MINgetZjlxDP():
    url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=1.000001&secid2=0.399001&' + \
          'fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&' + \
          'ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery18308174687833149541_1607783437004&_=1607783437202'
    res = requests.get(url)

    try:
        data1 = json.loads(res.text[41:-2])['data']['klines']
    except:
        return pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1],columns=['datetime', 'boss', 'small', 'med', 'big', 'huge'])
    min.drop(labels=['small','med','big','huge'],axis=1,inplace=True)
    # min['datetime'] = min['datetime'].astype('datetime64[ns]')
    min['boss'] = min['boss'].astype('float')
    min['net'] = min['boss'] - min['boss'].shift(1)
    min['bossma5'] = min['boss'].rolling(5).mean()

    return min

def calAmtFactor(n):
    global tdxdata

    # df_day = pd.DataFrame(api.get_index_bars(9, 1, '999999', 0, n+1))
    df_day = tdxdata.get_kline_data('999999',0,n+1,9)

    if 'date' not in df_day.columns:
        df_day['date'] = df_day['datetime'].apply(lambda x: x[:10])
    if 'amount' not in df_day.columns:
        df_day['amount'] = df_day['成交额']
    daylist = df_day['date'].values[:-1]

    times = (n+1)*240//800+1 if (n+1)*240%800>0 else (n+1)*240//800
    df_min = pd.DataFrame()
    for i in range(times-1,-1,-1):
        # temp = pd.DataFrame(api.get_index_bars(8, 1, '999999', i*800, 800))
        temp = tdxdata.get_kline_data('999999',i*800,800,8)
        if len(temp)>0:
            df_min = pd.concat([df_min, temp])

    if 'amount' not in df_min.columns:
        df_min['amount'] = df_min['成交额']
    df_min['date'] = df_min['datetime'].apply(lambda x: x[:10])
    df_min['time'] = df_min['datetime'].apply(lambda x: x[11:])
    df_min = df_min[(df_min['date']>=daylist[0]) & (df_min['date']<=daylist[-1])]
    df_min.reset_index(drop=True,inplace=True)

    ttt = df_min.pivot_table(index='time',values='amount',aggfunc='sum')
    ttt['cum'] = ttt['amount'].cumsum()
    ttt['ratio'] = ttt['cum'].values[-1] / ttt['cum']

    return list(ttt['ratio'].values)

def getSingleCCBData(tdxData, name, backset=0, klines=200, period=9):

    if period==0:
        code = etf_dict2[name]
    else:
        code = etf_dict2[name]
    df_single= tdxData.get_kline_data(code, backset=backset, klines=klines, period=period)
    df_single.reset_index(drop=True,inplace=True)
    if len(df_single)==0:
        print(f'getSingleCCBData {code} kline error,quitting')
        return
    # df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30') if x[-5:]=='13:00' else x)
    df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30'))
    # df_single['c6sig'] = six_pulse_excalibur(df_single)

    ccbcode = etf_ccb_dict[name]
    df_ccb =  tdxData.get_kline_data(ccbcode, backset=backset, klines=klines, period=period)
    if len(df_ccb)==0:
        print('getSingleCCBData {code} ccb error, quitting')
        return
    # df_ccb['ccb6sig'] = six_pulse_excalibur(df_ccb)
    df_ccb.rename(columns={'close':'ccb','high':'ccbh','low':'ccbl','open':'ccbo'},inplace=True)
    data = pd.merge(df_ccb[['datetime','ccb','ccbh','ccbl','ccbo']], df_single[['datetime','open','close','high','low','volume']], on='datetime',how='left')
    # data['gap6sig'] = data.apply(lambda x: x['c6sig']-x['ccb6sig'], axis=1)

    return data

def getBKZjlxRT(bkcode):

    url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=90.' + bkcode + \
          '&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56&ut=fa5fd1943c7b386f172d6893dbfba10b&cb=jQuery112406142175621622367_1615545163205&_=1615545163206'
    res = requests.get(url)

    try:
        data1 = json.loads(res.text[42:-2])['data']['klines']
    except:
        return pd.DataFrame({'datetime':['2000-01-01'], 'boss':[0]})
    min = pd.DataFrame([i.split(',') for i in data1],columns=['datetime', 'boss', 'small', 'med', 'big', 'huge'])
    min.drop(labels=['small','med','big','huge'],axis=1,inplace=True)
    # min['time'] = min['time'].astype('datetime64[ns]')
    min['boss'] = min['boss'].astype('float')/100000000

    return min

def getDPdata():

    dp_zjlx = MINgetZjlxDP()


    dp_zjlx['datetime'] = dp_zjlx['datetime'].apply(lambda x: x.replace('13:00','11:30'))

    # datetime,boss,net,bossma5
    dp_index, dp_preclose = MINgetDPindex()  # ['datetime', 'amount', 'vol', 'close', 'allamt', 'avg', 'amttrend']
    dp_index['datetime'] = dp_index['datetime'].apply(lambda x: x.replace('13:00', '11:30'))

    # df_poscnt = tdxdata.get_kline_data('880005',0,241,8)
    #
    # if len(df_poscnt) > 0:
    #     df_poscnt.rename(columns={'close':'upcnt'}, inplace=True)
    #     dp_zjlx = pd.merge(dp_zjlx, df_poscnt[['datetime','upcnt']], on='datetime',how='left')
    # else:
    #     dp_zjlx['upcnt'] = 0

    df_dp = pd.merge(dp_index, dp_zjlx, on='datetime',how='left')

    return df_dp, dp_preclose

def getOptiondata():
    global backset, trendKline, png_dict, tdxdata, df_optlist, new_optlist,kline_dict,etf_dict2

    periodkey = '1分钟k线'
    period = int(kline_dict[periodkey])
    klines= 300

    df_options = pd.DataFrame()
    new_optlist = {}

    for k,v in etf_dict2.items():
        optLongCode = png_dict[k].split('\n')[0].split(':')[1].split('_')[0]
        optShortCode = png_dict[k].split('\n')[1].split(':')[1].split('_')[0]

        ETFprice = tdxdata.get_kline_data(v, backset=backset, klines=klines, period=period)['close'].values[-1]
        longrow = df_optlist.loc[df_optlist['code']==optLongCode]
        shortrow = df_optlist.loc[df_optlist['code']==optShortCode]
        
        df_long = tdxdata.get_kline_data(optLongCode, backset=backset, klines=klines, period=period)
        df_long['longm20'] = df_long['close'].rolling(20).mean()
        tmp = df_long[df_long['datetime'].str.contains('15:00')]
        if '15:00' in df_long['datetime'].values[-1]:
            preidx = tmp.index[-2]
        else:
            preidx = tmp.index[-1]
        df_long = df_long.iloc[preidx:]

        df_short = tdxdata.get_kline_data(optShortCode, backset=backset, klines=klines, period=period)
        df_short['shortm20'] = df_short['close'].rolling(20).mean()
        tmp = df_short[df_short['datetime'].str.contains('15:00')]
        if '15:00' in df_short['datetime'].values[-1]:
            preidx = tmp.index[-2]
        else:
            preidx = tmp.index[-1]
        df_short = df_short.iloc[preidx:]

        df_long.rename(columns={'close':'long'}, inplace=True)
        df_short.rename(columns={'close':'short'}, inplace=True)
        df_opt = pd.merge(df_long[['datetime','long','longm20']], df_short[['datetime','short','shortm20']], on='datetime',how='inner')

        df_opt['etf'] = k
        df_options = pd.concat([df_options, df_opt])

        long_itm = max(0, ETFprice - longrow['行权价'].values[0])
        long_otm = df_opt['long'].values[-1] - long_itm
        short_itm = max(0, shortrow['行权价'].values[0]-ETFprice)
        short_otm = df_opt['short'].values[-1] - short_itm

        longtext = f'''认购:{optLongCode}_{longrow['name'].values[0]}_{df_opt['long'].values[-1]:.4f}=itm{long_itm:.4f}+otm{long_otm:.4f}'''
        shorttext = f'''认沽:{optShortCode}_{shortrow['name'].values[0]}_{df_opt['short'].values[-1]:.4f}=itm{short_itm:.4f}+otm{short_otm:.4f}'''
        new_optlist[k] = f'''{longtext}\n{shorttext}'''


    opt_Pivot = df_options.pivot_table(index='datetime',columns='etf',values=['long', 'longm20','short','shortm20'], dropna=False)

    return opt_Pivot

def getETFdata():
    global backset, threshold_pct,bins,trade_rate,trendKline, png_dict,tdxdata

    periodkey = '1分钟k线'
    period = int(kline_dict[periodkey])
    klines= int(kline_qty[periodkey])

    df_all = pd.DataFrame()

    for k,v in etf_dict2.items():
        df_single = getSingleCCBData(tdxdata,k, backset=backset, klines=klines, period=period)
        df_BKzjlx = getBKZjlxRT(etfbk_dict[k])
        df_single = pd.merge(df_single, df_BKzjlx[['datetime','boss']], on='datetime',how='left')

        tmp = df_single[df_single['datetime'].str.contains('15:00')]
        if '15:00' in df_single['datetime'].values[-1]:
            preidx = tmp.index[-2]
        else:
            preidx = tmp.index[-1]
        preclose =   df_single.loc[preidx,'close']
        df_single['preclose'] = preclose
        # dp_h = max(preclose, df_single[preidx+1:].high.max())
        # dp_l = min(preclose, df_single[preidx+1:].low.min())
        # dp_hh = dp_l + (dp_h - dp_l) * 7.5 / 8
        # dp_ll = dp_l + (dp_h - dp_l) * 0.5 / 8
        # df_single.loc[(df_single.close < dp_hh) & (df_single.close.shift(1) > dp_hh), 'crossdw'] = df_single.close
        # df_single.loc[(df_single.close > dp_ll) & (df_single.close.shift(1) < dp_ll), 'crossup'] = df_single.close

        df_single['cp30'] = (df_single['close'] - df_single['close'].rolling(30).min()) / (
                    df_single['close'].rolling(30).max() - df_single['close'].rolling(30).min())
        df_single['cp60'] = (df_single['close'] - df_single['close'].rolling(60).min()) / (
                    df_single['close'].rolling(60).max() - df_single['close'].rolling(60).min())
        df_single['cm10'] = df_single['close'].rolling(10).mean()
        df_single['cabovem10'] = df_single['close'] > df_single['cm10']

        df_single.loc[(df_single['cp30'] < 0.3) & (df_single['cp60'] < 0.3) & (df_single['cabovem10'] == True) & \
                  (df_single['close'] > df_single['close'].shift(1)), 'pivotup'] = 0.5
        df_single.loc[(df_single['cp30'] > 0.7) & (df_single['cp60'] > 0.7) & (df_single['cabovem10'] == False) & \
                  (df_single['close'] < df_single['close'].shift(1)), 'pivotdw'] = 0.5

        df_single['cm5'] = df_single['close'].rolling(5).mean()
        df_single['cm10'] = df_single['close'].rolling(10).mean()
        df_single['cm20'] = df_single['close'].rolling(20).mean()

        df_single['chhv60'] = df_single['high'].rolling(60).max()
        df_single['cllv60'] = df_single['low'].rolling(60).min()
        df_single['ccp60'] = df_single.apply(lambda x: (x['close']-x['cllv60'])/(x['chhv60']-x['cllv60']), axis=1)

        df_single['bossm10'] = df_single['boss'].rolling(10).mean()
        df_single['ccbm5'] = df_single['ccb'].rolling(5).mean()
        # df_single['ccbm10'] = df_single['ccb'].rolling(10).mean()
        df_single['ccbm20'] = df_single['ccb'].rolling(20).mean()

        df_single.loc[(df_single['close']>df_single['close'].shift(1)) & (df_single['boss']>df_single['boss'].shift(1)), 'upp'] = df_single['boss']
        df_single.loc[(df_single['close']<df_single['close'].shift(1)) & (df_single['boss']<df_single['boss'].shift(1)), 'dww'] = df_single['boss']
        df_single.loc[(df_single['close']>df_single['cm10']) & (df_single['upp'].notnull()), 'bossup'] = 1
        df_single.loc[(df_single['close']<df_single['cm10']) & (df_single['dww'].notnull()), 'bossdw'] = 1
        df_single['bossflag'] = df_single.apply(lambda x: 1 if x['bossup'] == 1 else (-1 if x['bossdw'] == 1 else np.nan), axis=1)
        df_single['bossflag'] = df_single['bossflag'].ffill()
        df_single.loc[(df_single['bossflag']==1) & (df_single['bossflag'].shift(1)==-1), 'bosssigup'] = df_single['close']
        df_single.loc[(df_single['bossflag']==-1) & (df_single['bossflag'].shift(1)==1), 'bosssigdw'] = df_single['close']

        df_single['etf'] = k
        df_all = pd.concat([df_all, df_single])
    df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'close', 'high', 'low','open','preclose','volume',
               'cm5','cm20','ccp60','pivotup','pivotdw','bosssigup','bosssigdw','boss'], dropna=False)

    return df_pivot

def plot_morning(df):
    global dp_boss, dp_amount, dp_preclose, timetitle,seq

    df_plot = df.copy()
    df_plot.reset_index(drop=True, inplace=True)
    df_plot.reset_index(drop=False, inplace=True)

    if int(seq)<30:
        maxx = 60
    elif int(seq)<60:
        maxx = 90
    else:
        maxx = 120

    fig, axes = plt.subplots(3, 2, figsize=(16,10))
    for ax in axes[:,:1]:
        ax[0].set_xticks(np.arange(0, 121, 30))
        ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130'))
    for ax in axes[:,1:]:
        ax[0].set_xticks(np.arange(0, 121, 30))
        ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130'))

    axes[0][0].hlines(y=dp_preclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
    axes[0][0].plot(df_plot.index, df_plot['close'],  linewidth=1, color='red')
    axes[0][0].plot(df_plot.index, df_plot['cm20'],  linewidth=0.8, color='red',linestyle='--')
    axes[0][0].plot(df_plot.index, df_plot['avg'], linewidth=1, color='violet')

    ax00b = axes[0][0].twinx()
    ax00b.bar(df_plot.index, df_plot.allamt, label='amount', color='grey', alpha=0.3, zorder=-14)
    ax00b.set_yticks([])
    ax00c = axes[0][0].twinx()
    ax00c.plot(df_plot.index, df_plot.boss, color='blue', linewidth=1, alpha=0.7)

    ax00d = axes[0][0].twinx()
    ax00d.scatter(df_plot.index, df_plot['pivotup'], label='转折点',marker='^', s=49, c='red', alpha=0.6)
    ax00d.scatter(df_plot.index, df_plot['pivotdw'], label='转折点',marker='v', s=49, c='green', alpha=0.7)
    ax00d.scatter(df_plot.index, df_plot['crossup'], label='底部涨',marker='D', s=25, c='red', alpha=0.7)
    ax00d.scatter(df_plot.index, df_plot['crossdw'], label='顶部跌',marker='D', s=25, c='green', alpha=0.8)
    ax00d.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6, linestyle='--', zorder=-25)
    ax00d.plot(df_plot.index, df_plot.dp_6sig-3, color='darkviolet', lw=1, alpha=0.5)
    ax00d.set_ylim(-10, 10)
    ax00d.set_yticks([])

    func00 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x,(x/dp_preclose-1)*100)
    axes[0][0].yaxis.set_major_formatter(mtick.FuncFormatter(func00))

    axes[0][0].text(0.5, 1.02, f' {timetitle}',
             horizontalalignment='center', transform=axes[0][0].transAxes, fontsize=12, fontweight='bold', color='black')

    axes[0][0].minorticks_on()
    axes[0][0].grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
    axes[0][0].grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

    if int(seq) > 90:
        ax00d.legend(loc='upper left',framealpha=0.1)
        # ax00b[0].legend(loc='lower left',framealpha=0.1)
    else:
        ax00d.legend(loc='upper right', framealpha=0.1)

    axes[0][1].plot(df_plot.index, df_plot['close'],  label='上证', linewidth=1, color='red')
    ax01b = axes[0][1].twinx()
    ax01c = axes[0][1].twinx()
    ax01d = axes[0][1].twinx()
    ax01b.plot(df_plot.index, df_plot.amttrend, label='成交量', color='green', lw=1.5, alpha=0.5)
    ax01c.plot(df_plot.index, df_plot.boss, label='主力', color='blue', linewidth=1, alpha=0.7)
    ax01c.hlines(y=0, xmin=df_plot.index.min(), xmax=maxx, colors='blue', linestyles='--', lw=2, alpha=0.4,zorder=-20)
    ax01c.set_yticks([])

    ax01d.scatter(df_plot.index, df_plot['pivotup'], label='转折点',marker='^', s=49, c='red', alpha=0.6)
    ax01d.scatter(df_plot.index, df_plot['pivotdw'], label='转折点',marker='v', s=49, c='green', alpha=0.7)
    ax01d.scatter(df_plot.index, df_plot['crossup'], label='底部涨',marker='D', s=25, c='red', alpha=0.7)
    ax01d.scatter(df_plot.index, df_plot['crossdw'], label='顶部跌',marker='D', s=25, c='green', alpha=0.8)
    ax01d.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6, linestyle='--', zorder=-25)
    ax01d.plot(df_plot.index, df_plot.dp_6sig-3, label='六脉', color='darkviolet', lw=1, alpha=0.5)
    ax01d.set_ylim(-10, 10)
    ax01d.set_yticks([])

    axes[0][1].text(0.5, 1.02, f'主力资金(蓝线):{dp_boss:.0f}亿 成交量(绿线):{dp_amount:.0f}亿',
             horizontalalignment='center', transform=axes[0][1].transAxes, fontsize=12, fontweight='bold', color='black')

    axes[0][1].minorticks_on()
    axes[0][1].grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
    axes[0][1].grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

    func01 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x,(x/dp_preclose-1)*100)
    axes[0][1].yaxis.set_major_formatter(mtick.FuncFormatter(func01))

    # axes[0][1].legend(loc='upper right',framealpha=0.1)
    if int(seq) < 90:
        ax01b.legend(loc='lower right',framealpha=0.1)
        ax01c.legend(loc='center right',framealpha=0.1)
    else:
        ax01b.legend(loc='lower left',framealpha=0.1)
        ax01c.legend(loc='center left',framealpha=0.1)

    keylist =  list(etf_dict.keys())
    funcx0 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[0])].values[1] - 1) * 100)
    funcx1 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[1])].values[1] - 1) * 100)
    funcx2 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[2])].values[1] - 1) * 100)
    funcx3 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[3])].values[1] - 1) * 100)

    for i,k in enumerate(etf_dict.keys()):

        if i==0:
            x = axes[1][0]
        elif i==1:
            x = axes[1][1]
        elif i==2:
            x = axes[2][0]
        else:
            x = axes[2][1]
        lastclose = df_plot[('preclose', k)].values[1]
        pct = df_plot[('close',k)].dropna().values[-1] / lastclose*100 - 100
        x.hlines(y=lastclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
        x.scatter(df_plot.index, df_plot[('bosssigup', k)], s=25, c='r', label='资金up',marker='o', alpha=0.8, zorder=-30)
        x.scatter(df_plot.index, df_plot[('bosssigdw', k)], s=25, c='g', label='资金dw',marker='o', alpha=0.8, zorder=-30)

        x.plot(df_plot.index, df_plot[('close', k)], linewidth=1, linestyle='-', color='red', alpha=1.)
        # x.plot(df_plot.index, df_plot[('cm5', k)], label='ma5', linewidth=0.7, linestyle='-', color='red', alpha=1.)
        x.plot(df_plot.index, df_plot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='--', color='red', alpha=1.)
        # x.vlines(openbar, ymin=df_plot[('close', k)].min(), ymax=df_pivot[('close', k)].max(), color='blue', linestyles='--',alpha=1)
        x.plot(df_plot.index, df_plot[f'avg_{k}'], linewidth=1, color='violet')

        x3 = x.twinx()
        x3.scatter(df_plot.index, df_plot[('pivotup', k)], label='转折点',s=25, c='r', marker='^', alpha=0.7,zorder=-10)
        x3.scatter(df_plot.index, df_plot[('pivotdw', k)], label='转折点',s=25, c='g', marker='v', alpha=0.7,zorder=-10)
        x3.scatter(df_plot.index, df_plot[f'crossup_{k}'], label='底部涨',s=16, c='r', marker='D', alpha=0.7,zorder=-10)
        x3.scatter(df_plot.index, df_plot[f'crossdw_{k}'], label='顶部跌',s=16, c='g', marker='D', alpha=0.7,zorder=-10)
        x3.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k',linewidth=0.5, alpha=0.6, zorder=-25)
        x3.plot(df_plot.index, df_plot[f'{k}_6sig'] - 3, color='darkviolet', lw=1, alpha=0.5)
        x3.set_ylim(-10, 10)
        x3.set_yticks([])

        x4 = x.twinx()
        x4.plot(df_plot.index, df_plot[('ccb', k)], linewidth=0.9, linestyle='-', color='green')
        # x4.plot(df_plot.index, df_plot[('ccbm20', k)], linewidth=0.6, linestyle='-.', color='green')
        x4.set_yticks([])

        x5 = x.twinx()
        x5.bar(df_plot.index, df_plot[('volume', k)], color='gray', alpha=0.3, zorder=-15)
        x5.set_yticks([])

        x6 = x.twinx()
        x6.plot(df_plot.index, df_plot[('boss', k)], linewidth=0.6, linestyle='-', color='blue')
        x6.hlines(y=0, xmin=df_plot.index.min(), xmax=maxx, colors='blue', linestyles='--', lw=2, alpha=0.4,zorder=-20)
        # x6.set_yticks([])

        if int(seq) < 90:
            x.legend(loc='upper right',framealpha=0.1)
            x3.legend(loc='center right',framealpha=0.1)
        else:
            x.legend(loc='upper left',framealpha=0.1)
            x3.legend(loc='center left',framealpha=0.1)

        if i==0:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx0))
        elif i==1:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx1))
        elif i==2:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx2))
        else:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx3))

        x.minorticks_on()
        x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

        if k in png_dict.keys():
            x.text(0.35,0.95,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')
        x.text(0.9, 1.02, f'涨跌:{pct:.2f}%',
                 horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    # plt.suptitle(timetitle,x=0.6, y=0.98)
    plt.savefig(f'output\\持续监控全景_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_1H.png')
    # plt.savefig(f'output\\持续监控全景_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_{seq}.png')

    fig.clf()
    plt.close(fig)


def plot_fullday(df):
    global dp_boss, dp_amount, dp_preclose, timetitle,seq

    df_plot = df.copy()
    df_plot.reset_index(drop=True, inplace=True)
    df_plot.reset_index(drop=False, inplace=True)

    if int(seq)<150:
        maxx = 150
    elif int(seq)<180:
        maxx = 180
    else:
        maxx = 240

    fig, axes = plt.subplots(5, 1, figsize=(12, 10.5))
    for ax in axes:
        ax.set_xticks(np.arange(0, 241, 30))
        ax.set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500'))

    axes[0].hlines(y=dp_preclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
    axes[0].plot(df_plot.index, df_plot['close'], linewidth=1, color='red')
    axes[0].plot(df_plot.index, df_plot['cm20'], linewidth=0.8, color='red', linestyle='--')
    axes[0].plot(df_plot.index, df_plot['avg'], linewidth=1, color='violet')

    ax0b = axes[0].twinx()
    ax0c = axes[0].twinx()
    ax0d = axes[0].twinx()
    ax0e = axes[0].twinx()

    ax0b.plot(df_plot.index, df_plot.boss, label='主力资金', color='blue', linewidth=1, alpha=0.7)
    ax0b.set_yticks([])
    ax0c.bar(df_plot.index, df_plot.allamt, label='amount', color='grey', alpha=0.3, zorder=-14)
    ax0c.set_yticks([])
    ax0d.plot(df_plot.index, df_plot.amttrend, label='成交量', color='green', lw=1.5, alpha=0.5)

    ax0e.scatter(df_plot.index, df_plot['pivotup'], label='转折点', marker='^', s=49, c='red', alpha=0.6)
    ax0e.scatter(df_plot.index, df_plot['pivotdw'], label='转折点', marker='v', s=49, c='green', alpha=0.7)
    ax0e.scatter(df_plot.index, df_plot['crossup'], label='底部涨', marker='D', s=25, c='red', alpha=0.7)
    ax0e.scatter(df_plot.index, df_plot['crossdw'], label='顶部跌', marker='D', s=25, c='green', alpha=0.8)
    ax0e.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6, linestyle='--', zorder=-25)
    ax0e.plot(df_plot.index, df_plot.dp_6sig-3, color='darkviolet', lw=1, alpha=0.5)
    ax0e.set_ylim(-10, 10)
    ax0e.set_yticks([])
    axes[0].text(0.5, 1.02, f'大盘资金(蓝线):{dp_boss:.0f}亿 成交量(绿线):{dp_amount:.0f}亿  {timetitle}',
                 horizontalalignment='center', transform=axes[0].transAxes, fontsize=12, fontweight='bold',
                 color='black')

    funcax0 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x,(x/dp_preclose-1)*100)
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(funcax0))

    axes[0].minorticks_on()
    axes[0].grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
    axes[0].grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

    # axes[0].legend(loc='upper left', framealpha=0.1)
    ax0e.legend(loc='upper left', framealpha=0.1)
    ax0b.legend(loc='lower left', framealpha=0.1)

    keylist =  list(etf_dict.keys())
    funcx0 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[0])].values[1] - 1) * 100)
    funcx1 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[1])].values[1] - 1) * 100)
    funcx2 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[2])].values[1] - 1) * 100)
    funcx3 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[3])].values[1] - 1) * 100)

    for i, k in enumerate(etf_dict.keys()):

        x = axes[i + 1]
        lastclose = df_plot[('preclose', k)].values[1]
        pct = df_plot[('close',k)].dropna().values[-1] / lastclose*100 - 100

        x.hlines(y=lastclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
        x.scatter(df_plot.index, df_plot[('bosssigup', k)], s=25, c='r', label='资金up', marker='o', alpha=0.8,
                  zorder=-30)
        x.scatter(df_plot.index, df_plot[('bosssigdw', k)], s=25, c='g', label='资金dw', marker='o', alpha=0.8,
                  zorder=-30)

        x.plot(df_plot.index, df_plot[('close', k)], linewidth=1, linestyle='-', color='red', alpha=1.)
        # x.plot(df_plot.index, df_plot[('cm5', k)], label='ma5', linewidth=0.7, linestyle='-', color='red', alpha=1.)
        x.plot(df_plot.index, df_plot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='--', color='red', alpha=1.)
        # x.vlines(openbar, ymin=df_plot[('close', k)].min(), ymax=df_pivot[('close', k)].max(), color='blue', linestyles='--',alpha=1)
        x.plot(df_plot.index, df_plot[f'avg_{k}'], linewidth=1, color='violet')

        # funcx = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', k)].values[1] - 1) * 100)
        if i==0:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx0))
        elif i==1:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx1))
        elif i==2:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx2))
        else:
            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx3))


        x3 = x.twinx()
        x3.scatter(df_plot.index, df_plot[('pivotup', k)], label='转折点', s=25, c='r', marker='^', alpha=0.7, zorder=-10)
        x3.scatter(df_plot.index, df_plot[('pivotdw', k)], label='转折点', s=25, c='g', marker='v', alpha=0.7, zorder=-10)
        x3.scatter(df_plot.index, df_plot[f'crossup_{k}'], s=16, c='r', marker='D', alpha=0.7, zorder=-10)
        x3.scatter(df_plot.index, df_plot[f'crossdw_{k}'], s=16, c='g', marker='D', alpha=0.7, zorder=-10)
        x3.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6,
                  zorder=-25)
        x3.plot(df_plot.index, df_plot[f'{k}_6sig']-3, color='darkviolet', lw=1, alpha=0.5)
        x3.set_ylim(-10, 10)
        x3.set_yticks([])

        x4 = x.twinx()
        x4.plot(df_plot.index, df_plot[('ccb', k)], linewidth=0.9, linestyle='-', color='green')
        # x4.plot(df_plot.index, df_plot[('ccbm20', k)], linewidth=0.6, linestyle='-.', color='green')
        x4.set_yticks([])

        x5 = x.twinx()
        x5.bar(df_plot.index, df_plot[('volume', k)], color='gray', alpha=0.3, zorder=-15)
        x5.set_yticks([])

        x6 = x.twinx()
        x6.plot(df_plot.index, df_plot[('boss', k)], linewidth=0.6, linestyle='-', color='blue')
        x6.hlines(y=0, xmin=df_plot.index.min(), xmax=maxx, colors='blue', linestyles='--',  lw=2, alpha=0.4,zorder=-20)
        # x6.set_yticks([])

        # if int(seq)>210:
        x.legend(loc='upper left', framealpha=0.1)
        x3.legend(loc='lower left', framealpha=0.1)

        x.minorticks_on()
        x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

        if k in png_dict.keys():
            x.text(0.25, 0.9, png_dict[k], horizontalalignment='center', transform=x.transAxes, fontsize=12,
                   fontweight='bold', color='black')
        x.text(0.7, 1.02, f'{k} 涨跌:{pct:.2f}%',
                 horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(f'output\\持续监控全景_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_2H.png')
    # plt.savefig(f'output\\持续监控全景_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_{seq}.png')

    fig.clf()
    plt.close(fig)

def plotAll():
    global dp_boss, dp_amount, dp_preclose,timetitle,seq

    df_dapan,dp_preclose = getDPdata()
    df_etf1min = getETFdata()

    df_etf1min.reset_index(drop=False, inplace=True)
    df_etf1min['datetime'] = df_etf1min['datetime'].apply(lambda x: x.replace('13:00','11:30'))
    # df_etf1min.set_index('datetime', inplace=True)

    df_all = pd.merge(df_dapan, df_etf1min, on='datetime', how='left')
    df_all['dp_6sig'] = analyzer(df_all).six_pulse_excalibur()
    for k, v in etf_dict2.items():
        tmp = df_all[['datetime', ('close',k), ('high',k),('low',k),('open',k),('preclose',k)]]
        tmp.columns = ['datetime','close', 'high', 'low', 'open', 'preclose']
        df_all[f'{k}_6sig'] = analyzer(tmp).six_pulse_excalibur()

    #for n in range(239,240,1):
    if True:
        # df_temp = df_all[:100+1]
        df_temp = df_all
        dp_h = max(dp_preclose, df_temp.close.max())
        dp_l = min(dp_preclose, df_temp.close.min())
        dp_hh = dp_l + (dp_h - dp_l) * 7.5 / 8
        dp_ll = dp_l + (dp_h - dp_l) * 0.5 / 8
        df_temp.loc[(df_temp.close < dp_hh) & (df_temp.close.shift(1) > dp_hh), 'crossdw'] = -0.5
        df_temp.loc[(df_temp.close > dp_ll) & (df_temp.close.shift(1) < dp_ll), 'crossup'] = -0.5

        df_temp['cp30'] = (df_temp['close'] - df_temp['close'].rolling(30).min()) / (
                    df_temp['close'].rolling(30).max() - df_temp['close'].rolling(30).min())
        df_temp['cp60'] = (df_temp['close'] - df_temp['close'].rolling(60).min()) / (
                    df_temp['close'].rolling(60,min_periods=31).max() - df_temp['close'].rolling(60,min_periods=31).min())
        df_temp['cm10'] = df_temp['close'].rolling(10).mean()
        df_temp['cm20'] = df_temp['close'].rolling(20).mean()
        df_temp['cabovem10'] = df_temp['close'] > df_temp['cm10']

        df_temp.loc[(df_temp['cp30'] < 0.3) & (df_temp['cp60'] < 0.3) & (df_temp['cabovem10'] == True) & \
                  (df_temp['close'] > df_temp['close'].shift(1)), 'pivotup'] = 0.5
        df_temp.loc[(df_temp['cp30'] > 0.7) & (df_temp['cp60'] > 0.7) & (df_temp['cabovem10'] == False) & \
                  (df_temp['close'] < df_temp['close'].shift(1)), 'pivotdw'] = 0.5

        for k, v in etf_dict2.items():
            preclose = df_temp[('preclose',k)].values[-1]
            k_h = max(preclose, df_temp[('high',k)].max())
            k_l = min(preclose, df_temp[('low',k)].min())
            k_hh = k_l + (k_h - k_l) * 7.5 / 8
            k_ll = k_l + (k_h - k_l) * 0.5 / 8
            df_temp[f'crossdw_{k}'] = np.nan
            df_temp[f'crossup_{k}'] = np.nan
            df_temp.loc[(df_temp[('close',k)] < k_hh) & (df_temp[('close',k)].shift(1) > k_hh), f'crossdw_{k}'] = -0.5
            df_temp.loc[(df_temp[('close',k)] > k_ll) & (df_temp[('close',k)].shift(1) < k_ll), f'crossup_{k}'] = -0.5

            tmp=df_temp[[('close',k),('volume',k)]]
            tmp['volcum'] = tmp[('volume',k)].cumsum()
            tmp['amt'] = tmp[('close',k)]*tmp[('volume',k)]
            tmp['amtcum'] = tmp['amt'].cumsum()
            tmp[f'avg_{k}'] = tmp['amtcum']/tmp['volcum']
            df_temp[f'avg_{k}'] = tmp[f'avg_{k}']

        seq = str(len(df_temp)).zfill(3)
        ktime = df_temp['datetime'].values[-1][2:].replace('-','')
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        timetitle = f'K线:{ktime}  - 时间戳 {stamp}'

        dp_boss = df_temp['boss'].ffill().values[-1]/100000000
        dp_amount = df_temp['amttrend'].ffill().values[-1]/100000000


        if len(df_temp) <= 120:
            df_temp = pd.concat([pd.DataFrame([[]])*1,df_temp])
            df_plot = pd.concat([df_temp, pd.DataFrame([[]] * (121 - len(df_temp)))])
            plot_morning(df_plot)
        elif len(df_temp)>120 and len(df_temp)<=240:
            df_temp = pd.concat([pd.DataFrame([[]])*1, df_temp])
            df_plot = pd.concat([df_temp, pd.DataFrame([[]] * (241 - len(df_temp)))])
            plot_fullday(df_plot)
        else:
            print('df>240')

def plot_options():

    global df_optlist, new_optlist

    df_opt = getOptiondata()
    df_opt = df_opt#[:100]
    df_opt.reset_index(drop=False, inplace=True)
    if 'index' in df_opt.columns:
        del df_opt['index']
    df_opt.reset_index(drop=True, inplace=True)
    df_opt.reset_index(drop=False, inplace=True)

    keylist =  list(etf_dict.keys())
    funcx00 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('long', keylist[0])].values[0] - 1) * 100)
    funcx01 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('short', keylist[0])].values[0] - 1) * 100)
    funcx10 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('long', keylist[1])].values[0] - 1) * 100)
    funcx11 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('short', keylist[1])].values[0] - 1) * 100)
    funcx20 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('long', keylist[2])].values[0] - 1) * 100)
    funcx21 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('short', keylist[2])].values[0] - 1) * 100)
    funcx30 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('long', keylist[3])].values[0] - 1) * 100)
    funcx31 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x*10000, (x / df_opt[('short', keylist[3])].values[0] - 1) * 100)

    timestamp = df_opt['datetime'].values[-1].replace('-','')

    if len(df_opt)<=120:

        maxx = 60 if len(df_opt)<45 else 120
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for ax in axes[:, :1]:
            ax[0].set_xticks(np.arange(0, 121, 30))
            ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130'))
        for ax in axes[:, 1:]:
            ax[0].set_xticks(np.arange(0, 121, 30))
            ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130'))

        for i, k in enumerate(etf_dict.keys()):
            if i == 0:
                x = axes[0][0]
            elif i == 1:
                x = axes[0][1]
            elif i == 2:
                x = axes[1][0]
            else:
                x = axes[1][1]


            x.plot(df_opt.index, df_opt[('long', k)], linewidth=0.6, label='认购(左)', linestyle='-', color='red')
            x.plot(df_opt.index, df_opt[('longm20', k)], linewidth=0.6, linestyle='--', color='red')
            x.hlines(y=df_opt[('long', k)].values[0], xmin=df_opt.index.min(), xmax=maxx, colors='red', lw=1, alpha=0.5,zorder=-20)

            x1 = x.twinx()
            x1.plot(df_opt.index, df_opt[('short', k)], linewidth=0.6, label='认沽(右)',linestyle='-', color='green')
            x1.plot(df_opt.index, df_opt[('shortm20', k)], linewidth=0.6, linestyle='--', color='green')
            x1.hlines(y=df_opt[('short', k)].values[0], xmin=df_opt.index.min(), xmax=maxx, colors='green', lw=1, alpha=0.5,zorder=-20)

            if k in png_dict.keys():
                x.text(0.5, 0.95, new_optlist[k], horizontalalignment='center', transform=x.transAxes, fontsize=12,
                       fontweight='bold', color='black')

            if i == 0:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx00))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx01))
            elif i == 1:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx10))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx11))
            elif i == 2:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx20))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx21))
            else:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx30))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx31))

            x.minorticks_on()
            x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
            x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

            x.legend(loc='center left', fontsize=10, frameon=True, framealpha=0.1)
            x1.legend(loc='center right', fontsize=10, frameon=True, framealpha=0.1)

        plt.tight_layout()
        plt.suptitle(timestamp,x=0.5, y=0.99)
        plt.savefig(f'output\\持续监控_期权_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_1H.png')
        # plt.savefig(f'output\\持续监控_期权_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_{seq}.png')

        fig.clf()
        plt.close(fig)

    else:

        maxx = 180 if len(df_opt) < 180 else 240
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        for ax in axes:
            ax.set_xticks(np.arange(0, 241, 30))
            ax.set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500'))

        for i, k in enumerate(etf_dict.keys()):
            x = axes[i]

            x.plot(df_opt.index, df_opt[('long', k)], linewidth=0.6, label='认购(左)', linestyle='-', color='red')
            x.plot(df_opt.index, df_opt[('longm20', k)], linewidth=0.6, linestyle='--', color='red')
            x.hlines(y=df_opt[('long', k)].values[0], xmin=df_opt.index.min(), xmax=maxx, colors='red', lw=1, alpha=0.5,zorder=-20)

            x1 = x.twinx()
            x1.plot(df_opt.index, df_opt[('short', k)], linewidth=0.6, label='认沽(右)',linestyle='-', color='green')
            x1.plot(df_opt.index, df_opt[('shortm20', k)], linewidth=0.6, linestyle='--', color='green')
            x1.hlines(y=df_opt[('short', k)].values[0], xmin=df_opt.index.min(), xmax=maxx, colors='green', lw=1, alpha=0.5,zorder=-20)

            if k in png_dict.keys():
                x.text(0.35, 0.92, new_optlist[k], horizontalalignment='center', transform=x.transAxes, fontsize=12,
                       fontweight='bold', color='black')

            if i == 0:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx00))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx01))
            elif i == 1:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx10))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx11))
            elif i == 2:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx20))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx21))
            else:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx30))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx31))

            x.minorticks_on()
            x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
            x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

            x.legend(loc='center left', fontsize=10, frameon=True, framealpha=0.1)
            x1.legend(loc='center right', fontsize=10, frameon=True, framealpha=0.1)

        plt.tight_layout()
        plt.suptitle(timestamp,x=0.7, y=0.98)
        plt.savefig(f'output\\持续监控_期权_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_2H.png')
        # plt.savefig(f'output\\持续监控_期权_v2.8_{datetime.datetime.now().strftime("%Y%m%d")}_{seq}.png')
        fig.clf()
        plt.close(fig)

    return



def main():

    global factor, dayr1,png_dict, tdxdata, df_optlist

    if (time.strftime("%H%M", time.localtime()) > '0900' and time.strftime("%H%M", time.localtime()) <= '0930'):
        print('waiting market, sleep 40s')
        time.sleep(40)

    try:

        while (time.strftime("%H%M", time.localtime())>='0930' and time.strftime("%H%M", time.localtime())<='1502'):

            if (time.strftime("%H%M", time.localtime())>'1130' and time.strftime("%H%M", time.localtime())<'1300'):
                print('sleep 60s')
                time.sleep(60)
            else:
                try:
                    png_dict,df_optlist = getMyOptions()
                except:
                    png_dict = {}
                plotAll()
                if plotopt == 'Y':
                    plot_options()
                time.sleep(30)

        plotAll()
        if plotopt == 'Y':
            plot_options()

        return
    except Exception as e:
        print('exception msg: '+ str(e))
        print(' *****  exception, recreate tdxdata then restart main ***** ')
        tdxdata.api.close()
        tdxdata.Exapi.close()
        tdxdata = mytdxData()
        time.sleep(5)
        main()


if __name__ == '__main__':

    prog_start = time.time()
    print('-------------------------------------------')
    print('Job start !!! ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))

    cfg_fn = 'monitor_v2.8.cfg'
    config = configparser.ConfigParser()
    config.read(cfg_fn, encoding='utf-8')
    dte_low = int(dict(config.items('option_screen'))['dte_low'])
    dte_high = int(dict(config.items('option_screen'))['dte_high'])
    close_Threshold = float(dict(config.items('option_screen'))['close_threshold'])
    etf_ccb_dict = dict(config.items('etf_ccb_dict'))
    etfbk_dict = dict(config.items('etfbk_dict'))
    etf_dict = dict(config.items('etf_dict'))
    etf_dict2 = dict(config.items('etf_dict2'))
    etfcode_dict = dict(config.items('etfcode_dict'))
    kline_dict = dict(config.items('kline_dict'))
    kline_qty = dict(config.items('kline_qty'))
    backset = int(dict(config.items('backset'))['backset'])
    png_dict = dict(config.items('png_dict'))
    etf_threshold = dict(config.items('etf_threshold'))
    opt_path = dict(config.items('path'))['opt_path']
    pushurl = dict(config.items('pushmessage'))['url']
    plotopt = dict(config.items('plotimgs'))['option']


    tdxdata  = mytdxData()

    now = tdxdata.get_kline_data('399001',0,20,8)
    tempdates = tdxdata.get_kline_data('399001',0,20,9)
    opt_fn_last =  opt_path +  '\\沪深期权清单_'+ tempdates['datetime'].values[-2][:10].replace('-','')+'.csv'
    opt_fn =  opt_path +  '\\沪深期权清单_'+ now['datetime'].values[-1][:10].replace('-','')+'.csv'
    print(opt_fn)

    try:
        png_dict,df_optlist = getMyOptions()
    except:
        png_dict = {}

    factor = calAmtFactor(5)
    factor = factor+[1.00]

    # plotAll()
    # if plotopt == 'Y':
    #     plot_options()

    main()

    tdxdata.api.close()
    tdxdata.Exapi.close()

    time_end = time.time()
    print('-------------------------------------------')
    print(f'Job completed!!!  All time costed: {(time_end - prog_start):.0f}秒')
