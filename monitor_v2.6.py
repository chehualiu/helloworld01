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


if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('output'):
    os.makedirs('output')

import logging
# 创建logger对象
logger = logging.getLogger('mylogger')
# 设置日志等级
logger.setLevel(logging.DEBUG)
# 追加写入文件a ，设置utf-8编码防止中文写入乱码
test_log = logging.FileHandler('logs\\monitor_v2.6_'+datetime.datetime.now().strftime('%Y%m%d')+ \
                               '.log','a',encoding='gbk')
# 向文件输出的日志级别
test_log.setLevel(logging.INFO)
# 向文件输出的日志信息格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message).300s',datefmt='%m-%d %H:%M:%S')
# formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
test_log.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

# 加载文件到logger对象中
logger.addHandler(test_log)
logger.addHandler(console)

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

        if code.isdigit() and len(code)==6: # A股
            if code[:2] in ['15','00','30','16','12','39','18']: # 深市
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

def six_pulse_excalibur(df):

    CLOSE=df['close']
    LOW=df['low']
    HIGH=df['high']
    DIFF=EMA(CLOSE,8)-EMA(CLOSE,13)
    DEA=EMA(DIFF,5)
    ABC1=IF(DIFF>DEA,1,0)

    zjsc1=(CLOSE-LLV(LOW,8))/(HHV(HIGH,8)-LLV(LOW,8))*100
    K=SMA(zjsc1,3,1)
    D=SMA(K,3,1)
    ABC2=IF(K>D,1,0)

    ydzb=REF(CLOSE,1)
    RSI1=(SMA(MAX(CLOSE-ydzb,0),5,1))/(SMA(ABS(CLOSE-ydzb),5,1))*100
    RSI2=(SMA(MAX(CLOSE-ydzb,0),13,1))/(SMA(ABS(CLOSE-ydzb),13,1))*100
    ABC3=IF(RSI1>RSI2,1,0)

    zjsc=(HHV(HIGH,13)-CLOSE)/(HHV(HIGH,13)-LLV(LOW,13))*100
    LWR1=SMA(zjsc,3,1)
    LWR2=SMA(LWR1,3,1)
    ABC4=IF(LWR1>LWR2,1,0)

    BBI=(MA(CLOSE,3)+MA(CLOSE,5)+MA(CLOSE,8)+MA(CLOSE,13))/4
    ABC5=IF(CLOSE>BBI,1,0)

    MTM=CLOSE-REF(CLOSE,1)
    MMS=100*EMA(EMA(MTM,5),3)/EMA(EMA(ABS(MTM),5),3)
    MMM=100*EMA(EMA(MTM,13),8)/EMA(EMA(ABS(MTM),13),8)
    ABC6=IF(MMS>MMM,1,0)

    signal=ABC1+ABC2+ABC3+ABC4+ABC5+ABC6

    return signal


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
            logger.info('reusing option file')
            data = pd.read_csv(opt_fn, encoding='gbk',dtype={'ETFcode':str,'code':str})
        else:
            try:
                data = getAllOptionsV3()
                logger.info('New option file')
                data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
            except:
                logger.info('update failed, reusing option file')
                data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    else:
        logger.info('New option file ' + opt_fn)
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


    return png_dict

def original_strings(x):
    if x.dtype == 'object':
        return ', '.join(x)
    else:
        return x.iloc[0]


def readDapanData(startdate,enddate):

    dtype_dic = {'open':float,'high':float,'low':float,'close':float,'volume':float,
                 'amount':float,'turn':float,'pctChg':float}
    df_sh = pd.read_csv('D:\\stockstudy\\win2021\\baodata\\bao-sh.000001.csv',dtype=dtype_dic, encoding='gbk', )
    df_sh = df_sh[(df_sh['date']>=startdate) & (df_sh['date']<=enddate)]
    df_sz = pd.read_csv('D:\\stockstudy\\win2021\\baodata\\bao-sz.399001.csv',dtype=dtype_dic, encoding='gbk', )
    df_sz = df_sz[(df_sh['date']>=startdate) & (df_sz['date']<=enddate)]

    return df_sh,df_sz

def getSouthzjlx():

    url = 'https://push2.eastmoney.com/api/qt/kamtbs.rtmin/get?'

    params={
        'cb': 'jQuery11230987190397908317_1725940063923',
        'fields1': 'f1,f2,f3,f4',
        'fields2': 'f51,f54,f52,f58,f53,f62,f56,f57,f60,f61',
        'ut': 'b2884a393a59ad64002292a3e90d46a5',
        '_': '1725940063945',
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'Referer': 'https://data.eastmoney.com/hsgt/hsgtV2.html',
        # 'Accept-Encoding': 'gzip, deflate,br,zstd',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'Host': 'push2.eastmoney.com'}

    res = requests.get(url=url,params=params,headers=headers)

    try:
        data1 = json.loads(res.text[41:-2])['data']['n2s']
    except:
        return pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1],columns=['time', 'hgtnet', 'hgtin', 'sgtnet', 'hgtout', 'south', 'sgtin','sgtout', 'northin', 'northout'])
    min = min[30:]
    min.reset_index(drop=True, inplace=True)
    min.drop(labels=['time', 'hgtnet', 'hgtin', 'sgtnet', 'hgtout', 'sgtin','sgtout', 'northin', 'northout'],axis=1,inplace=True)
    min = min[min['south']!='-']
    min['south'] = min['south'].astype('float')/10000
    min['southdelta'] = min['south'] - min['south'].shift(1)

    return min

def getNorthdata():

    url = 'https://dataq.10jqka.com.cn/fetch-data-server/fetch/v1/interval_data'

    params={ }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0',
        'Referer': 'https://data.eastmoney.com/hsgt/hsgtV2.html',
        'Content-Type': 'application/json;charset=utf-8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'Host': 'dataq.10jqka.com.cn'}

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    start = int(datetime.datetime.strptime(f'{today} 09:30', '%Y-%m-%d %H:%M').timestamp())    # "1728351000"
    end = int(datetime.datetime.strptime(f'{today} 15:00', '%Y-%m-%d %H:%M').timestamp())
    data = {"indexes":[{"codes":["48:883957"],"index_info":[{"index_id":"hsgt_main_money"}]}],
            "time_range":{"time_type":"TREND", "start": start, "end": end}}

    res = requests.post(url=url,params=params,headers=headers, json=data)

    try:
        data1 = json.loads(res.text)['data']['data'][0]['values'][0]['values']
    except:
        return pd.DataFrame()
    min = pd.DataFrame(data1,columns=['north'])
    # min.reset_index(drop=True, inplace=True)
    min['northdelta'] = min['north'] - min['north'].shift(1)

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
    min['boss'] = min['boss'].astype('float')/100000000
    min['net'] = min['boss'] - min['boss'].shift(1)
    min['bossma5'] = min['net'].rolling(5).mean()

    return min

def MINgetDPindexOld():
    global factor

    url = 'http://push2his.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery112409887318613050171_1615514818501&secid=1.000001&secid2=0.399001'+\
          '&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6%2Cf7%2Cf8%2Cf9%2Cf10%2Cf11&fields2=f51%2Cf53%2Cf56%2Cf58&iscr=0&ndays=1&_=1615514818608'
    res = requests.get(url)
    try:
        data1 = json.loads(res.text[42:-2])['data']['trends']
        preclose = json.loads(res.text[42:-2])['data']['preClose']
    except:
        return pd.DataFrame()
    data = pd.DataFrame([i.split(',') for i in data1], columns=['time', 'close', 'vol', 'mean'])
    data.drop(labels=['mean'],axis=1,inplace=True)

    data['close'] = data['close'].astype('float')
    data['vol'] = data['vol'].astype('float')
    data['amt'] = data['close']*data['vol']
    data['amtcum'] = data['amt'].cumsum()
    data['volcum'] = data['vol'].cumsum()
    data['avg'] = data['amtcum']/data['volcum']
    data['amttrend'] = 0

    for i in range(len(data)):
        data['amttrend'][i] = data['amtcum'][i] * factor[i]
    data.drop(columns=['amtcum','volcum','vol','amt'], inplace=True)

    data['close'] = data['close'].astype('float')
    data['avg'] = data['avg'].astype('float')

    return data, preclose

def MINgetDPindex():
    global factor

    # day_sh =  pd.DataFrame(api.get_index_bars(9, 1, '999999', 0, 30))
    day_sh =  tdxdata.get_kline_data('999999',0,30,9)
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

    data = pd.merge(df_sh[['time','amount','vol','close']], df_sz[['time','amountsz']], how='inner')
    data['allamt'] = data['amount']+data['amountsz']
    data['amt'] = data['close']*data['vol']
    data['amtcum'] = data['amt'].cumsum()
    data['volcum'] = data['vol'].cumsum()
    data['avg'] = data['amtcum']/data['volcum']
    data['amt'] = data['amount'] + data['amountsz']
    data['amtcum'] = data['amt'].cumsum()
    data['factor'] = factor[:len(data)]
    data['amttrend'] = data['factor']*data['amtcum']
    data.iloc[-1, data.columns.get_loc('amttrend')] = np.nan

    data.drop(columns=['amtcum','volcum','amt','factor'], inplace=True)

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
        tempstr = re.search(r'\(({.*})\)', res.text).group(1)
        data1 = json.loads(tempstr)['data']['trends']
        preclose = json.loads(tempstr)['data']['preClose']
    except:
        return pd.DataFrame(),0

    data = pd.DataFrame([i.split(',') for i in data1], columns=['time', 'ref1', 'close', 'ref2','ref3', 'ref4','vol', 'avg'])
    data.drop(labels=['ref1', 'ref2','ref3', 'ref4'],axis=1,inplace=True)
    data['time'] = data['time'].astype('datetime64[ns]')
    data['close'] = data['close'].astype('float')
    data['vol'] = data['vol'].astype('float')
    data['avg'] = data['avg'].astype('float')

    day = datetime.datetime.now().strftime('%Y-%m-%d')
    #df_poscnt = pd.DataFrame(api.get_index_bars(8,1, '880005', 0, 240))
    df_poscnt = tdxdata.get_kline_data('880005',0,240,8)
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


def plotAllzjlx():
    df_north = getNorthdata()
    # df_south = getSouthzjlx()
    # df_southNorth = pd.merge(df_north, df_south, left_index=True, right_index=True)
    hs300_zjlx = getHS300zjlx()  # zjlxdelta
    hs300_index,hs_preclose = getETFindex('510300')  #   close
    if len(hs300_index) == 0:
        logger.info('failed to get 300ETF df, skipped')
        return
    dp_zjlx = MINgetZjlxDP()   # zjlxdelta
    dp_index, dp_preclose = MINgetDPindex()  # close

    dp_boss = dp_zjlx.boss.values[-1]/100000000
    # north_amt = df_north.north.values[-1]
    north_amt = df_north.north.values[-1]/100000000
    hs300_boss = hs300_zjlx.boss.values[-1]
    upcnt = hs300_index.upcnt.ffill().values[-1]
    # df_dp = pd.merge(df_north, dp_zjlx, left_index=True, right_index=True)
    # df_dp = pd.merge(df_southNorth, dp_zjlx, left_index=True, right_index=True)
    df_dp = pd.merge(df_north, dp_zjlx, left_index=True, right_index=True)
    df_dp = pd.merge(df_dp, dp_index, left_index=True, right_index=True)

    dp_h = max(dp_preclose, dp_index.close.max())
    dp_l = min(dp_preclose, dp_index.close.min())
    dp_hh = dp_l + (dp_h-dp_l)*7.5/8
    dp_ll = dp_l + (dp_h-dp_l)*0.5/8
    df_dp.loc[(df_dp.close<dp_hh) & (df_dp.close.shift(1)>dp_hh), 'crossdw'] = df_dp.close
    df_dp.loc[(df_dp.close>dp_ll) & (df_dp.close.shift(1)<dp_ll), 'crossup'] = df_dp.close

    df_hs300 = pd.merge(hs300_zjlx, hs300_index,left_index=True, right_index=True)
    hs_h = max(hs_preclose, df_hs300.close.max())
    hs_l = min(hs_preclose, df_hs300.close.min())
    hs_hh = hs_l + (hs_h-hs_l)*7.5/8
    hs_ll = hs_l + (hs_h-hs_l)*0.5/8
    df_hs300.loc[(df_hs300.close<hs_hh) & (df_hs300.close.shift(1)>hs_hh), 'crossdw'] = df_hs300.close
    df_hs300.loc[(df_hs300.close>hs_ll) & (df_hs300.close.shift(1)<hs_ll), 'crossup'] = df_hs300.close


    dp_amount = str(int(dp_index['amttrend'].values[-2]/100000000))+'亿'

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
        logger.info('UP 510300 hs300' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = pushurl + msg
        requests.get(msgURL)

    if df_hs300['dw'].values[-1]==True:
        msg = 'UP 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))
        logger.info('DOWN 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = pushurl + msg
        requests.get(msgURL)

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

    df_dp['up'] = df_dp.apply(lambda x: x.close if x.up==True else np.nan, axis=1)
    df_dp['dw'] = df_dp.apply(lambda x: x.close if x.dw==True else np.nan, axis=1)
    if len(df_dp)<240:
        df_dp = pd.concat([df_dp, pd.DataFrame([[]]*(240-len(df_dp)))])
    df_dp.reset_index(drop=True,inplace=True)
    df_dp.reset_index(inplace=True)
    df_dp.iloc[0:2, df_dp.columns.get_loc('amttrend')] = np.nan

    #################

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,6), dpi=100)
    ax1.set_xticks(np.arange(0, 241, 30))
    ax1.set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330','1400','1430','1500'))
    ax2.set_xticks(np.arange(0, 241, 30))
    ax2.set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330','1400','1430','1500'))
    ax1b = ax1.twinx()
    ax1c = ax1.twinx()
    ax1d = ax1.twinx()

    ax2b = ax2.twinx()
    ax2c = ax2.twinx()
    ax2d = ax2.twinx()

    df_dp.plot(x='index', y='close', label='dp', linewidth=1, color='red', ax=ax1,zorder=10)
    df_dp.plot(x='index', y='avg', linewidth=1, markersize=10, color='violet', label='avg', ax=ax1,zorder=11)
    ax1.scatter(df_dp.index, df_dp['up'], marker='^', s=100, c='red',alpha=0.4)
    ax1.scatter(df_dp.index, df_dp['dw'], marker='v',s=100, c='green',alpha=0.5)
    ax1.scatter(df_dp.index, df_dp['crossup'], marker='D', s=36, c='red',alpha=0.4)
    ax1.scatter(df_dp.index, df_dp['crossdw'], marker='D',s=36, c='green',alpha=0.5)

    ax1.hlines(y=dp_preclose, xmin=0, xmax=240-3, colors='aqua', linestyles='-', lw=2, label='preclose')

    ax1b.bar(df_dp.index, df_dp.net, label='dpzjlx', color='blue', alpha=0.2, zorder=-15)
    # ax1c.bar(df_dp.index, df_dp.northdelta, label='north', color='grey', alpha=0.5, zorder=-14)
    ax1c.bar(df_dp.index, df_dp.allamt, label='amount', color='grey', alpha=0.3, zorder=-14)
    ax1b.plot(df_dp.index, df_dp.bossma5, label='zjlxma5', color='blue', lw=0.5)

    ax1c.hlines(y=0, xmin=0, xmax=240-3, colors='black', linestyles='-', lw=0.3)
    ax1d.plot(df_dp.index, df_dp.amttrend, label='amttrend', color='green', lw=1.5, alpha=0.5)
    # ax1.text(0.5,0.95,' 大盘主力资金(蓝柱):' + str(round(dp_boss,2)) + ' 北向流入(灰柱):' + str(round(north_amt,2)) + ' 量能(绿线):'+ dp_amount,
    ax1.text(0.5,0.90,f'大盘主力资金(蓝柱):{dp_boss:.2f} 量能(绿线):{dp_amount}',
             horizontalalignment='center',transform=ax1.transAxes, fontsize=12, fontweight='bold', color='black')

    df_hs300.plot(x='index', y='close', label='hs300', linewidth=1, color='red', ax=ax2,zorder=10)
    df_hs300.plot(x='index', y='avg', linewidth=1, markersize=10, color='violet', label='avg', ax=ax2,zorder=11)
    ax2.hlines(y=hs_preclose, xmin=0, xmax=240-3, colors='aqua', linestyles='-', lw=2, label='preclose')
    ax2.scatter(df_hs300.index, df_hs300['up'], marker='^', s=100, c='red',alpha=0.4)
    ax2.scatter(df_hs300.index, df_hs300['dw'], marker='v',s=100, c='green',alpha=0.5)
    ax2.scatter(df_hs300.index, df_hs300['crossup'], marker='D', s=36, c='red',alpha=0.4)
    ax2.scatter(df_hs300.index, df_hs300['crossdw'], marker='D',s=36, c='green',alpha=0.5)

    # ax2b.bar(df_hs300.index, df_hs300.net, label='zjlx', color='blue', alpha=0.2, zorder=-15)
    # ax2b.plot(df_hs300.index, df_hs300.bossma5, label='zjlxm5', color='blue', lw=0.5)
    ax2b.plot(df_hs300.index, df_hs300.boss, label='hs300boss', color='blue', lw=0.8, zorder=-17,alpha=0.8)
    ax2c.plot(df_hs300.index, df_hs300.upcnt, label='upcnt', color='green', lw=1.5,alpha=0.5 )
    ax2d.bar(df_dp.index, df_dp.northdelta, label=None, color='blue', alpha=0.2, zorder=-14)
    ax2d.hlines(y=0, xmin=0, xmax=240-3, colors='black', linestyles='-', lw=0.3)
    ax2.text(0.5,0.90, f'HS300 主力流入(蓝线):{hs300_boss:.0f} 北向流入(蓝柱):{north_amt:.1f}亿 上涨数(绿线): {upcnt:.0f}',
             horizontalalignment='center',transform=ax2.transAxes, fontsize=12, fontweight='bold', color='black')
    ax1b.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax1c.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax2b.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax2d.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)

    func1 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x,(x/dp_preclose-1)*100)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(func1))
    func2 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x,(x/hs_preclose-1)*100)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(func2))

    ax1.minorticks_on()
    ax1.grid(which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    ax1.set(xlabel=None)

    ax2.minorticks_on()
    ax2.grid(which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    ax2.set(xlabel=None)

    plt.suptitle('DP HS300 - 时间戳 ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
    plt.tight_layout()

    plt.savefig('output\\持续监控DP_HS300_v2.6_'+ datetime.datetime.now().strftime('%Y%m%d') + '.png' )

    fig.clf()
    plt.close(fig)
    return

def get_xdxr_EM(code):
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
    df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30') if x[-5:]=='13:00' else x)
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



def construct_ohlc_collections(data, wid=0.4,linewidths=0.8,marketcolors=None, config=None):

    open_price = data['open'].values
    close_price = data['close'].values
    high_price = data['high'].values
    low_price = data['low'].values

    colors = (data['close']>data['open']).map({True:'red',False:'green'})
    facecolors = (data['close']>data['open']).map({True:'white',False:'green'})

    HLsegments = [[(i, low_price[i]), (i, high_price[i])] for i in range(len(open_price))]
    openSegments = [[(i-wid, open_price[i]), (i, open_price[i])] for i in range(len(open_price))]
    closeSegments = [[(i, close_price[i]), (i+wid, close_price[i])] for i in range(len(open_price))]

    line_segments1 = LineCollection(HLsegments, linewidths=linewidths, colors=colors)
    line_segments2 = LineCollection(openSegments, linewidths=linewidths*2, colors=colors)
    line_segments3 = LineCollection(closeSegments, linewidths=linewidths*2, colors=colors)

    return line_segments1, line_segments2, line_segments3

def getKlineObjects(data, linewidths=0.8, bar_width=0.3, bar_linewidth=1.5):

    # Sample data
    open_price = data['open'].values
    close_price = data['close'].values
    high_price = data['high'].values
    low_price = data['low'].values

    segments1 = [[(i, low_price[i]), (i, min(open_price[i],close_price[i]))] for i in range(len(open_price))]
    segments2 = [[(i, max(open_price[i],close_price[i])),(i, high_price[i])] for i in range(len(open_price))]
    # segments2 = [[(i, low_price[i]), (i, high_price[i])] for i in range(len(open_price))]
    colors = (data['close']>data['open']).map({True:'red',False:'green'})
    facecolors = (data['close']>data['open']).map({True:'white',False:'green'})

    line_segments1 = LineCollection(segments1, linewidths=linewidths,colors=colors)
    line_segments2 = LineCollection(segments2, linewidths=linewidths,colors=colors)

    bar_centers = [i for i in range(len(open_price))]
    bar_bottoms = [open_price[i] if open_price[i] <= close_price[i] else close_price[i] for i in range(len(open_price))]
    bar_heights = [abs(open_price[i] - close_price[i]) for i in range(len(open_price))]
    bar_segments = PolyCollection([[(x - bar_width * 1, bottom), (x - bar_width * 1, bottom + height),
                                    (x + bar_width * 1, bottom + height), (x + bar_width * 1, bottom)]
                                   for x, bottom, height in zip(bar_centers, bar_bottoms, bar_heights)],
                                  facecolors=facecolors, edgecolors=colors)
    bar_segments.set_linewidth(bar_linewidth)
                                  # facecolors=facecolors, edgecolors=['green' if open_price[i] <= close_price[i]
                                  #                                    else 'red' for i in range(len(open_price))])
    return line_segments1,line_segments2,bar_segments


def getAllCCBmin1A():
    global backset, threshold_pct,bins,trade_rate,trendKline, png_dict,tdxdata

    periodkey = '1分钟k线'
    period = int(kline_dict[periodkey])
    klines= int(kline_qty[periodkey])

    df_all = pd.DataFrame()

    for k,v in etf_dict2.items():
        df_single = getSingleCCBData(tdxdata,k, backset=backset, klines=klines, period=period)
        tmp = df_single[df_single['datetime'].str.contains('15:00')]
        if '15:00' in df_single['datetime'].values[-1]:
            preidx = tmp.index[-2]
        else:
            preidx = tmp.index[-1]
        preclose =   df_single.loc[preidx,'close']
        dp_h = max(preclose, df_single[preidx+1:].high.max())
        dp_l = min(preclose, df_single[preidx+1:].low.min())
        dp_hh = dp_l + (dp_h - dp_l) * 7.5 / 8
        dp_ll = dp_l + (dp_h - dp_l) * 0.5 / 8
        df_single.loc[(df_single.close < dp_hh) & (df_single.close.shift(1) > dp_hh), 'crossdw'] = df_single.close
        df_single.loc[(df_single.close > dp_ll) & (df_single.close.shift(1) < dp_ll), 'crossup'] = df_single.close

        df_single['cp30'] = (df_single['close'] - df_single['close'].rolling(30).min()) / (
                    df_single['close'].rolling(30).max() - df_single['close'].rolling(30).min())
        df_single['cp60'] = (df_single['close'] - df_single['close'].rolling(60).min()) / (
                    df_single['close'].rolling(60).max() - df_single['close'].rolling(60).min())
        df_single['cm10'] = df_single['close'].rolling(10).mean()
        df_single['cabovem10'] = df_single['close'] > df_single['cm10']

        df_single.loc[(df_single['cp30'] < 0.3) & (df_single['cp60'] < 0.3) & (df_single['cabovem10'] == True) & \
                  (df_single['close'] > df_single['close'].shift(1)), 'up3'] = 1
        df_single.loc[(df_single['cp30'] > 0.7) & (df_single['cp60'] > 0.7) & (df_single['cabovem10'] == False) & \
                  (df_single['close'] < df_single['close'].shift(1)), 'dw3'] = 1


        # df_single.sort_values(by=['datetime'],ascending=True, inplace=True)
        df_single.reset_index(drop=True, inplace=True)
        df_single['pctChg'] = df_single['close']/df_single['close'].shift(1)-1

        df_single['close'] = df_single['close'].ffill()

        df_single['cm5'] = df_single['close'].rolling(5).mean()
        df_single['cm20'] = df_single['close'].rolling(20).mean()
        df_single['cmgap'] = (df_single['cm5'] - df_single['cm20'])/df_single['cm5']

        df_single['gap'] = (df_single['close'] - df_single['cm20'])/df_single['cm20']*100
        df_single['gapabs'] = df_single['gap'].apply(lambda x: abs(x))

        gap_threshold = float(etf_threshold[k])
        df_single.loc[(df_single['gap']>gap_threshold),'gapSig'] = df_single['gap']
        df_single.loc[(df_single['gap']<-1*gap_threshold),'gapSig'] = df_single['gap']


        df_single['chhv60'] = df_single['high'].rolling(60).max()
        df_single['cllv60'] = df_single['low'].rolling(60).min()
        df_single['ccp60'] = df_single.apply(lambda x: (x['close']-x['cllv60'])/(x['chhv60']-x['cllv60']), axis=1)

        df_single['ccbm5'] = df_single['ccb'].rolling(5).mean()
        df_single['ccbm20'] = df_single['ccb'].rolling(20).mean()
        df_single['ccbmgap'] = (df_single['ccbm5'] - df_single['ccbm20'])/df_single['ccbm5']

        df_single.loc[df_single['cmgap']<0,'cmark'] = -1
        df_single.loc[df_single['cmgap']>0,'cmark'] = 1
        df_single.loc[df_single['ccbmgap']<0,'ccbmark'] = 1
        df_single.loc[df_single['ccbmgap']>0,'ccbmark'] = -1
        df_single['mark'] = df_single['ccbmark'] + df_single['cmark']

        df_single['ccbhhv60'] = df_single['ccbh'].rolling(60).max()
        df_single['ccbllv60'] = df_single['ccbl'].rolling(60).min()
        df_single['ccbcp60'] = df_single.apply(lambda x: (x['ccb']-x['ccbllv60'])/(x['ccbhhv60']-x['ccbllv60']), axis=1)
        df_single['ccbgap'] = df_single['ccp60']-df_single['ccbcp60']

        df_single['ccbgapm20'] = df_single['ccbgap'].rolling(10).mean()
        df_single.loc[(df_single['ccbgap']>df_single['ccbgapm20']) & (df_single['mark']>=0),'up2'] = 0  # (df_single['mark']>=0) &
        df_single.loc[(df_single['ccbgap']<df_single['ccbgapm20']) & (df_single['mark']<=0),'dw2'] = 0

        # df_single.loc[(df_single['gap6sig']>=3) & (df_single['gap6sig'].shift(1)<=df_single['gap6sig']),'up'] = 0  # (df_single['mark']>=0) &
        # df_single.loc[(df_single['gap6sig']<=-3) & (df_single['gap6sig'].shift(1)>=df_single['gap6sig']),'dw'] = 0  # (df_single['mark']>=0) &

        df_single['etf'] = k
        df_all = pd.concat([df_all, df_single])

    df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'ccbh', 'ccbl','ccbo', 'close', 'high', 'low','open','volume','ccbm5','ccbm20',
               'gap','gapSig','cm5','cm20','cmgap','ccp60','ccbcp60','ccbgap','ccbgapm20','ccbmgap','up2','dw2','crossup','crossdw','up3','dw3'], dropna=False)
    df_pivot.reset_index(drop=False,inplace=True)

    df_pivot['time'] = df_pivot[('datetime','')].apply(lambda x: x.split(' ')[1])
    firstbar = df_pivot[('time','')].tolist().index('15:00')
    df_pivot = df_pivot[firstbar:]

    df_pivot = df_pivot[-241:]

    df_pivot.reset_index(drop=True,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)


    openbar = [id for id,v in enumerate(df_pivot[('datetime')].tolist()) if '09:31' in v][0]-1

    df_pivot['zero'] = 0
    lastBar = df_pivot[('datetime','')].values[-1].replace('-','').replace(':','').replace(' ','_')

    if plot1min == 'N':
        df_pivot = df_pivot[-121:]
        df_pivot.reset_index(drop=True, inplace=True)
        df_pivot.reset_index(drop=False, inplace=True)
        return df_pivot

    fig, ax = plt.subplots(4, 1, figsize=(18,9), sharex=True)

    tickGap = 30
    xlables = [i[5:].replace('-','').replace(':','').replace(' ','_') for i in df_pivot[( 'datetime','')][::tickGap]]
    plt.xticks(range(len(df_pivot))[::tickGap],xlables,rotation=90)

    for x,k in zip(ax,  etf_dict.keys()):



        dftemp = df_pivot[[('index', ''), ('open', k), ('close', k), ('high', k), ('low', k)]]
        dftemp.columns = ['index', 'open', 'close', 'high', 'low']

        line_seg1, line_seg2, line_seg3 = construct_ohlc_collections(dftemp, wid=0.4)
        x.add_collection(line_seg1)
        x.add_collection(line_seg2)
        x.add_collection(line_seg3)

        x.plot(df_pivot.index, df_pivot[('cm5', k)], label='ma5', linewidth=0.7, linestyle='-', color='red', alpha=1.)
        x.plot(df_pivot.index, df_pivot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='-.', color='red', alpha=1.)
        x.vlines(openbar, ymin=df_pivot[('close', k)].min(), ymax=df_pivot[('close', k)].max(), color='blue', linestyles='--',alpha=1)
        x.scatter(df_pivot.index, df_pivot[('crossup', k)], s=25, c='r', marker='D', alpha=0.8,zorder=-10)
        x.scatter(df_pivot.index, df_pivot[('crossdw', k)], s=25, c='g', marker='D', alpha=0.8,zorder=-10)

        x3 = x.twinx()
        # x3.plot(df_pivot.index, df_pivot[('gap6sig', k)], label='6pulse', linewidth=0.7, color='black', alpha=1, zorder=-10)
        # x3.scatter(df_pivot.index, df_pivot[('up', k)], marker='^', s=36, c='red',alpha=0.3,zorder=-30)
        # x3.scatter(df_pivot.index, df_pivot[('dw', k)], marker='v',s=36, c='green',alpha=0.3,zorder=-30)
        x3.scatter(df_pivot.index, df_pivot[('up2', k)], label='up',s=25, c='r', marker='s', alpha=0.3,zorder=-20)
        x3.scatter(df_pivot.index, df_pivot[('dw2', k)], label='dw',s=25, c='g', marker='s', alpha=0.3,zorder=-20)
        x3.scatter(df_pivot.index, df_pivot[('up3', k)], s=25, c='r', marker='^', alpha=0.7,zorder=-10)
        x3.scatter(df_pivot.index, df_pivot[('dw3', k)], s=25, c='g', marker='v', alpha=0.7,zorder=-10)
        # x3.plot(df_pivot.index, df_pivot[('gap', k)], label='m20gap',linewidth=0.5, color='violet', linestyle='-', zorder=-28, alpha=1)
        # x3.plot(df_pivot.index, df_pivot[('gapSig', k)], label='gapsig',linewidth=3, color='violet', zorder=-30, alpha=0.7)
        x3.hlines(float(etf_threshold[k]), xmin=df_pivot.index.min(), xmax=df_pivot.index.max(), color='violet',
                  linewidth=0.5, alpha=1.0, zorder=-25)
        x3.hlines(-1 * float(etf_threshold[k]), xmin=df_pivot.index.min(), xmax=df_pivot.index.max(), color='violet',
                  linewidth=0.5, alpha=1.0, zorder=-25)
        x3.set_ylim(-10, 10)
        # x3.plot(df_pivot.index, df_pivot[('ccbgap', k)], label='cpgap', linewidth=0.7, linestyle='-', color='blue')
        # x3.plot(df_pivot.index, df_pivot[('ccbgapm20', k)], label='cpgapma10', linewidth=0.7, linestyle='dotted', color='blue')

        x4 = x.twinx()
        df_tmp = df_pivot[[('ccb', k), ('ccbh', k), ('ccbl', k), ('ccbo', k)]].copy()
        df_tmp.columns = ['close', 'high', 'low', 'open']
        ccbline_seg1, ccbline_seg2, ccbbar_segments = getKlineObjects(df_tmp, linewidths=1.2, bar_width=0.2)
        x4.add_collection(ccbline_seg1)
        x4.add_collection(ccbline_seg2)
        x4.add_collection(ccbbar_segments)
        x4.plot(df_pivot.index, df_pivot[('ccbm5', k)], linewidth=0.9, linestyle='dotted', color='green')
        x4.plot(df_pivot.index, df_pivot[('ccbm20', k)], linewidth=0.6, linestyle='-.', color='green')
        x4.set_yticks([])

        x5 = x.twinx()
        x5.bar(df_pivot.index, df_pivot[('volume', k)], color='gray', alpha=0.3, zorder=-15)
        x5.set_yticks([])

        # x5 = x.twinx()
        # # x5.plot(df_pivot.index,df_pivot[('ccbgap',k)],  color='blue',linewidth=1.0,linestyle='-')  #marker='.',
        # # x5.plot(df_pivot.index,df_pivot[('ccbgapm20',k)], color='blue',linestyle='-', linewidth=0.6)
        # x5.scatter(df_pivot.index, df_pivot[('up', k)], marker='^', s=36, c='red',alpha=0.7,zorder=-30)
        # x5.scatter(df_pivot.index, df_pivot[('dw', k)], marker='v',s=36, c='green',alpha=0.7,zorder=-30)
        # x5.scatter(df_pivot.index, df_pivot[('up2', k)], s=25, c='r', marker='s', alpha=0.3,zorder=-20)
        # x5.scatter(df_pivot.index, df_pivot[('dw2', k)], s=25, c='g', marker='s', alpha=0.3,zorder=-20)
        # # x5.set_ylim(-2,2)
        # x5.set_yticks([])

        # x6 = x.twinx()
        # x6.plot(df_pivot.index,df_pivot[('gap',k)], linewidth=0.5, color='violet',linestyle='-',zorder=-28,alpha=1)
        # x6.plot(df_pivot.index,df_pivot[('gapSig',k)], linewidth=3, color='violet', zorder=-30,alpha=1)
        # x6.hlines(float(etf_threshold[k]), xmin=df_pivot.index.min(), xmax=df_pivot.index.max(), color='violet', linewidth=0.5, alpha=1.0, zorder=-25)
        # x6.hlines(-1* float(etf_threshold[k]), xmin=df_pivot.index.min(), xmax=df_pivot.index.max(), color='violet', linewidth=0.5, alpha=1.0, zorder=-25)
        # x6.set_yticks([])

        x.legend(loc='upper left')
        x3.legend(loc='center left')

        x.minorticks_on()
        x.grid(which='major', axis="x", color='k', linestyle='-', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='--', linewidth=0.15)

        if k in png_dict.keys():
            x.text(0.25,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        # x.text(0.1, 1., k, horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.suptitle(lastBar,x=0.6, y=0.98)
    plt.savefig('output\\持续监控ETF1分钟_v2.6_' + datetime.datetime.now().strftime('%Y%m%d') + '.png')

    fig.clf()
    plt.close(fig)

    df_pivot = df_pivot[-121:]

    df_pivot.reset_index(drop=True,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)

    return df_pivot


def getAllCCBmin5B():
    global backset, gaphist_pct,bins,png_dict

    periodkey = '5分钟k线'
    period = int(kline_dict[periodkey])
    klines= int(kline_qty[periodkey])

    df_all = pd.DataFrame()

    for k,v in etf_dict.items():
        df_single = getSingleCCBData(tdxdata,k,backset, klines,period)

        df_single['time'] = df_single['datetime'].apply(lambda x: x.split(' ')[1])
        df_single['pctChg'] = df_single['close']/df_single['close'].shift(1)-1

        df_single['cm5'] = df_single['close'].rolling(5).mean()
        df_single['cm20'] = df_single['close'].rolling(20).mean()
        df_single['cmgap'] = (df_single['cm5'] - df_single['cm20'])/df_single['cm5']
        df_single['chhv30'] = df_single['high'].rolling(30).max()
        df_single['cllv30'] = df_single['low'].rolling(30).min()
        df_single['ccp30'] = df_single.apply(lambda x: (x['close']-x['cllv30'])/(x['chhv30']-x['cllv30']), axis=1)

        df_single['gap'] = (df_single['close'] - df_single['cm20'])/df_single['close']*100
        df_single['gapabs'] = df_single['gap'].apply(lambda x: abs(x))
        gap_threshold = float(etf_threshold[k])
        df_single.loc[(df_single['gap']>gap_threshold),'gapSig'] = df_single['gap']
        df_single.loc[(df_single['gap']<-1*gap_threshold),'gapSig'] = df_single['gap']

        df_single['ccbma5'] = df_single['ccb'].rolling(5).mean()
        df_single['ccbma20'] = df_single['ccb'].rolling(20).mean()
        df_single['ccbmgap'] = (df_single['ccbma5'] - df_single['ccbma20'])/df_single['ccbma5']

        df_single.loc[df_single['cmgap']<0,'cmark'] = -1
        df_single.loc[df_single['cmgap']>0,'cmark'] = 1
        df_single.loc[df_single['ccbmgap']<0,'ccbmark'] = 1
        df_single.loc[df_single['ccbmgap']>0,'ccbmark'] = -1
        df_single['mark'] = df_single['ccbmark'] + df_single['cmark']

        df_single['ccbhhv30'] = df_single['ccbh'].rolling(30).max()
        df_single['ccbllv30'] = df_single['ccbl'].rolling(30).min()
        df_single['ccbcp30'] = df_single.apply(lambda x: (x['ccb']-x['ccbllv30'])/(x['ccbhhv30']-x['ccbllv30']), axis=1)
        df_single['ccbgap'] = df_single['ccp30']-df_single['ccbcp30']
        df_single['ccbgapm10'] = df_single['ccbgap'].rolling(10).mean()

        # df_single.loc[(df_single['gap6sig']>=2) & (df_single['gap6sig'].shift(1)<df_single['gap6sig']),'up'] = 0  # (df_single['mark']>=0) &
        # df_single.loc[(df_single['gap6sig']<=-2) & (df_single['gap6sig'].shift(1)>df_single['gap6sig']),'dw'] = 0  # (df_single['mark']>=0) &

        df_single.loc[(df_single['ccbgap']>df_single['ccbgapm10']) & (df_single['mark']>=0),'up2'] = 0  # (df_single['mark']>=0) &
        df_single.loc[(df_single['ccbgap']<df_single['ccbgapm10']) & (df_single['mark']<=0),'dw2'] = 0

        df_single.reset_index(drop=True,inplace=True)

        firstbar = df_single['time'].tolist().index('15:00')
        df_single = df_single[firstbar:]

        df_single['etf'] = k
        df_all = pd.concat([df_all, df_single])


    df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'ccbh', 'ccbl','ccbo', 'close', 'high', 'low','open','volume',
                               'pctChg', 'ccbma5', 'ccbma20', 'cm5', 'cm20', 'ccbgap','ccbgapm10','up2','dw2'], dropna=False)

    # df_pivot = df_pivot[-49:]
    df_pivot.reset_index(drop=False,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)

    if plot5min == 'N':
        df_pivot = df_pivot[-49:]
        if 'index' in df_pivot.columns:
            del df_pivot['index']
        df_pivot.reset_index(drop=True, inplace=True)
        df_pivot.reset_index(drop=False, inplace=True)
        return df_pivot

    lastBar = df_pivot[('datetime','')].values[-1][5:].replace('-','').replace(':','').replace(' ','_')
    openbar = [id for id,v in enumerate(df_pivot[('datetime')].tolist()) if '09:3' in v][-1]-1

    fig, ax = plt.subplots(len(etf_dict), 1, figsize=(10*1,9), sharex=True)

    tickGap = 24
    xlables = [i[5:].replace('-','').replace(':','').replace(' ','_') for i in df_pivot[( 'datetime','')][::tickGap]]
    plt.xticks(range(len(df_pivot))[::tickGap],xlables,rotation=90)

    for x,k in zip(ax,  etf_dict.keys()):

        dftemp = df_pivot[[('index', ''), ('open', k), ('close', k), ('high', k), ('low', k)]]
        dftemp.columns = ['index', 'open', 'close', 'high', 'low']
        line_seg1, line_seg2, line_seg3 = construct_ohlc_collections(dftemp, wid=0.4)
        x.add_collection(line_seg1)
        x.add_collection(line_seg2)
        x.add_collection(line_seg3)
        x.plot(df_pivot.index, df_pivot[('cm5', k)], label='ma5', linewidth=1, linestyle='dotted', color='red', alpha=1.)
        x.plot(df_pivot.index, df_pivot[('cm20', k)], label='ma20', linewidth=1, linestyle='-.', color='red', alpha=1.)
        x.vlines(openbar, ymin=df_pivot[('close', k)].min(), ymax=df_pivot[('close', k)].max(), color='blue',
                 linestyles='--', alpha=1)

        x2 = x.twinx()
        df_tmp = df_pivot[[('ccb', k), ('ccbh', k), ('ccbl', k), ('ccbo', k)]].copy()
        df_tmp.columns = ['close', 'high', 'low', 'open']
        ccbline_seg1, ccbline_seg2, ccbbar_segments = getKlineObjects(df_tmp, linewidths=1.2, bar_width=0.2)
        x2.add_collection(ccbline_seg1)
        x2.add_collection(ccbline_seg2)
        x2.add_collection(ccbbar_segments)
        x2.plot(df_pivot.index, df_pivot[('ccbma5', k)], label='ma5', linewidth=0.9, linestyle='dotted', color='green')
        x2.plot(df_pivot.index, df_pivot[('ccbma20', k)], label='ma20', linewidth=0.6, linestyle='-.', color='green')
        x2.set_yticks([])

        x3 = x.twinx()
        # x3.plot(df_pivot.index, df_pivot[('ccbgap', k)], color='blue', linewidth=0.7, linestyle='-')  # marker='.',
        # x3.plot(df_pivot.index, df_pivot[('ccbgapm10', k)], color='blue', linestyle='--', linewidth=0.5)
        # x3.set_yticks([])
        # x3.scatter(df_pivot.index, df_pivot[('up', k)], marker='^', s=36, c='red',alpha=0.3)
        # x3.scatter(df_pivot.index, df_pivot[('dw', k)], marker='v',s=36, c='green',alpha=0.3)
        x3.scatter(df_pivot.index, df_pivot[('up2', k)], s=25, c='r', marker='s', alpha=0.3,zorder=-20)
        x3.scatter(df_pivot.index, df_pivot[('dw2', k)], s=25, c='g', marker='s', alpha=0.3,zorder=-20)
        # x3.set_ylim(-2, 2)
        # x3.set_yticks([])

        x4 = x.twinx()
        x4.bar(df_pivot.index, df_pivot[('volume', k)], color='gray', alpha=0.3, zorder=-15)
        x4.set_yticks([])

        x.minorticks_on()
        x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.1)

        if k in png_dict.keys():
            x.text(0.25,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.suptitle(f'{df_pivot[("datetime","")].values[-1][-11:]}',x=0.6, y=0.98)

    plt.savefig('output\\持续监控ETF期权5分钟监控_v2.6_' + datetime.datetime.now().strftime('%Y%m%d') + '.png')
    fig.clf()
    plt.close(fig)

    df_pivot = df_pivot[-49:]
    if 'index' in df_pivot.columns:
        del df_pivot['index']
    df_pivot.reset_index(drop=True,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)

    return df_pivot

def drawAllCCBmin1A5B():
    global tdxdata

    df_1min = getAllCCBmin1A()
    df_5min = getAllCCBmin5B()

    # openbar = [id for id,v in enumerate(df_1min[('datetime')].tolist()) if '09:31' in v][0]-1
    # df_1min['zero'] = 0

    lastBar = df_1min[('datetime','')].values[-1].replace('-','').replace(':','').replace(' ','_')
    openbar = [id for id,v in enumerate(df_1min[('datetime')].tolist()) if ('09:31' in v or '13:01' in v)][0]-1
    openbar5m = [id for id,v in enumerate(df_5min[('datetime')].tolist()) if '09:35' in v][-1]-1

    nrows = 4
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(18,9),gridspec_kw={'width_ratios': [2, 4]})#, sharex=True)

    tickGap1min = 30
    xlables1min = [i[11:].replace('-','').replace(':','').replace(' ','_') for i in df_1min[( 'datetime','')][::tickGap1min]]

    tickGap5min = 6
    xlables5min = [i[8:].replace('-','').replace(':','').replace(' ','_') for i in df_5min[( 'datetime','')][::tickGap5min]]

    # plt.xticks(range(len(df_1min))[::tickGap1min],xlables1min,rotation=90)

    # 5分钟画图
    for x,k in zip(ax[:,:1].tolist(),  etf_dict.keys()):
        x  = x[0]
        dftemp = df_5min[[('index', ''), ('open', k), ('close', k), ('high', k), ('low', k)]]
        dftemp.columns = ['index', 'open', 'close', 'high', 'low']
        line_seg1, line_seg2, line_seg3 = construct_ohlc_collections(dftemp, wid=0.4)
        x.add_collection(line_seg1)
        x.add_collection(line_seg2)
        x.add_collection(line_seg3)
        x.plot(df_5min.index, df_5min[('cm5', k)], label='ma5', linewidth=1, linestyle='dotted', color='red', alpha=1.)
        x.plot(df_5min.index, df_5min[('cm20', k)], label='ma20', linewidth=0.7, linestyle='-.', color='red', alpha=1)
        x.vlines(openbar5m, ymin=df_5min[('close', k)].min(), ymax=df_5min[('close', k)].max(), color='blue',
                 linestyles='--', alpha=1)
        # x.set_yticks([])

        x2 = x.twinx()
        df_tmp = df_5min[[('ccb', k), ('ccbh', k), ('ccbl', k), ('ccbo', k)]].copy()
        df_tmp.columns = ['close', 'high', 'low', 'open']
        ccbline_seg1, ccbline_seg2, ccbbar_segments = getKlineObjects(df_tmp, linewidths=1.2, bar_width=0.2)
        x2.add_collection(ccbline_seg1)
        x2.add_collection(ccbline_seg2)
        x2.add_collection(ccbbar_segments)
        x2.plot(df_5min.index, df_5min[('ccbma5', k)], label='ma5', linewidth=0.9, linestyle='dotted', color='green')
        x2.plot(df_5min.index, df_5min[('ccbma20', k)], label='ma20', linewidth=0.6, linestyle='-.', color='green')
        x2.set_yticks([])

        x3 = x.twinx()
        # x3.scatter(df_5min.index, df_5min[('up', k)], marker='^', s=36, c='red',alpha=0.3)
        # x3.scatter(df_5min.index, df_5min[('dw', k)], marker='v',s=36, c='green',alpha=0.3)
        x3.scatter(df_5min.index, df_5min[('up2', k)], s=25, c='r', marker='s', alpha=0.3,zorder=-20)
        x3.scatter(df_5min.index, df_5min[('dw2', k)], s=25, c='g', marker='s', alpha=0.3,zorder=-20)
        x3.set_yticks([])

        x4 = x.twinx()
        x4.bar(df_5min.index, df_5min[('volume', k)], color='gray', alpha=0.3, zorder=-15)
        x4.set_yticks([])

        x.minorticks_on()
        x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.1)

        if k in png_dict.keys():
            x.text(0.4,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        x.set_xticks(range(len(df_5min))[::tickGap5min])
        x.set_xticklabels(xlables5min)

        # plt.suptitle(f'{df_5min[("datetime","")].values[-1][-11:]}',x=0.6, y=0.98)


    # 1分钟画图
    for x,k in zip(ax[:,1:],  etf_dict.keys()):
        x = x[0]

        dftemp = df_1min[[('index', ''), ('open', k), ('close', k), ('high', k), ('low', k)]]
        dftemp.columns = ['index', 'open', 'close', 'high', 'low']
        line_seg1, line_seg2, line_seg3 = construct_ohlc_collections(dftemp, wid=0.4)
        x.add_collection(line_seg1)
        x.add_collection(line_seg2)
        x.add_collection(line_seg3)

        x.plot(df_1min.index, df_1min[('cm5', k)], label='ma5', linewidth=0.7, linestyle='-', color='red', alpha=1.)
        x.plot(df_1min.index, df_1min[('cm20', k)], label='ma20', linewidth=0.7, linestyle='-.', color='red', alpha=1.)
        x.vlines(openbar, ymin=df_1min[('close', k)].min(), ymax=df_1min[('close', k)].max(), color='blue',
                 linestyles='--', alpha=1)
        x.scatter(df_1min.index, df_1min[('crossup', k)], s=25, c='r', marker='D', alpha=1,zorder=-10)
        x.scatter(df_1min.index, df_1min[('crossdw', k)], s=25, c='g', marker='D', alpha=1,zorder=-10)


        # x.vlines(openbar, ymin=df_1min[('close', k)].min(), ymax=df_1min[('close', k)].max(), color='blue', linestyles='--',alpha=1)

        x2 = x.twinx()
        df_tmp = df_1min[[('ccb', k), ('ccbh', k), ('ccbl', k), ('ccbo', k)]].copy()
        df_tmp.columns = ['close', 'high', 'low', 'open']
        ccbline_seg1, ccbline_seg2, ccbbar_segments = getKlineObjects(df_tmp, linewidths=1.2, bar_width=0.2)
        x2.add_collection(ccbline_seg1)
        x2.add_collection(ccbline_seg2)
        x2.add_collection(ccbbar_segments)
        x2.plot(df_1min.index, df_1min[('ccbm5', k)], label='ma5', linewidth=0.9, linestyle='dotted', color='green')
        x2.plot(df_1min.index, df_1min[('ccbm20', k)], label='ma20', linewidth=0.6, linestyle='-.', color='green')
        x2.set_yticks([])

        x3 = x.twinx()
        # x3.scatter(df_1min.index, df_1min[('up', k)], marker='^', s=36, c='red',alpha=0.3)
        # x3.scatter(df_1min.index, df_1min[('dw', k)], marker='v',s=36, c='green',alpha=0.3)
        x3.scatter(df_1min.index, df_1min[('up2', k)], label='up', s=25, c='r', marker='s', alpha=0.3,zorder=-20)
        x3.scatter(df_1min.index, df_1min[('dw2', k)], label='dw', s=25, c='g', marker='s', alpha=0.3,zorder=-20)
        x3.scatter(df_1min.index, df_1min[('up3', k)], s=25, c='r', marker='^', alpha=0.7,zorder=-10)
        x3.scatter(df_1min.index, df_1min[('dw3', k)], s=25, c='g', marker='v', alpha=0.7,zorder=-10)
        # x3.plot(df_1min.index,df_1min[('gap',k)], label='m20gap', linewidth=0.5, color='violet',linestyle='-',zorder=-28,alpha=1)
        # x3.plot(df_1min.index,df_1min[('gapSig',k)], label='gapSig', linewidth=3, color='violet', zorder=-30,alpha=0.7)
        x3.hlines(float(etf_threshold[k]), xmin=df_1min.index.min(), xmax=df_1min.index.max(), color='violet', linewidth=0.5, alpha=1.0, zorder=-25)
        x3.hlines(-1* float(etf_threshold[k]), xmin=df_1min.index.min(), xmax=df_1min.index.max(), color='violet', linewidth=0.5, alpha=1.0, zorder=-25)
        x3.set_ylim(-10, 10)
        x3.set_yticks([])
        # x3.plot(df_1min.index, df_1min[('ccbgap', k)], label='cpgap', linewidth=0.7, linestyle='-', color='blue')
        # x3.plot(df_1min.index, df_1min[('ccbgapm20', k)], label='cpgapma10', linewidth=0.7, linestyle='dotted', color='blue')

        x4 = x.twinx()
        x4.bar(df_1min.index, df_1min[('volume', k)], color='gray', alpha=0.3, zorder=-15)
        x4.set_yticks([])
        # x6 = x.twinx()
        # x6.plot(df_1min.index,df_1min[('gap',k)], linewidth=0.5, color='violet',linestyle='-',zorder=-28,alpha=1)
        # x6.plot(df_1min.index,df_1min[('gapSig',k)], linewidth=3, color='violet', zorder=-30,alpha=1)
        # x6.hlines(float(etf_threshold[k]), xmin=df_1min.index.min(), xmax=df_1min.index.max(), color='violet', linewidth=0.5, alpha=1.0, zorder=-25)
        # x6.hlines(-1* float(etf_threshold[k]), xmin=df_1min.index.min(), xmax=df_1min.index.max(), color='violet', linewidth=0.5, alpha=1.0, zorder=-25)
        # x6.set_yticks([])

        x.legend(loc='upper left')
        x3.legend(loc='center left')

        x.minorticks_on()
        x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.1)

        if k in png_dict.keys():
            x.text(0.4,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        # x.text(0.1, 1., k, horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        x.set_xticks(range(len(df_1min))[::tickGap1min])
        x.set_xticklabels(xlables1min)

    ax[0,1].text(0.8, 1.0, '信号: m5/m20+gapCP30', horizontalalignment='center', transform=ax[0,1].transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.suptitle(lastBar,x=0.35, y=0.98)
    plt.savefig('output\\持续监控ETF1A5B_v2.6_' + datetime.datetime.now().strftime('%Y%m%d') + '.png')

    fig.clf()
    plt.close(fig)

    if plotday == 'Y':
        df_all = pd.DataFrame()

        for k, v in etf_dict.items():
            df_single = getSingleCCBData(tdxdata, k, backset, 81, 9)
            df_single['name'] = k
            df_single['cm5'] = df_single['close'].rolling(5).mean()
            df_single['cm20'] = df_single['close'].rolling(20).mean()

            df_single['ccbm5'] = df_single['ccb'].rolling(5).mean()
            df_single['ccbm20'] = df_single['ccb'].rolling(20).mean()
            df_all = pd.concat([df_all, df_single])#[cols]])

        df_day = df_all.pivot_table(index=['datetime'],columns=['name'],values=['close','high','low','ccb','ccbh','ccbl',
                      'ccbm5','ccbm20','cm5','cm20'],dropna=False)

        nrows = 4
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols, figsize=(15,9),gridspec_kw={'width_ratios': [3, 4]})#, sharex=True)

        tickGapday = 20
        xlablesday = [i[:10].replace('-','') for i in df_day.index[::tickGapday]]

        tickGap5min = 6
        xlables5min = [i[8:].replace('-','').replace(' ','_') for i in df_5min[( 'datetime','')][::tickGap5min]]

        # plt.xticks(range(len(df_1min))[::tickGap1min],xlables1min,rotation=90)

        # 日K线画图
        for x,k in zip(ax[:,:1],  etf_dict.keys()):
            x = x[0]
            x.plot(df_day.index, df_day[('close', k)], label=k, linewidth=1, color='red', marker='.', markersize=4,
                   alpha=0.7)
            x.plot(df_day.index, df_day[('cm5', k)], linewidth=0.7, color='red', alpha=0.7, linestyle='-.')
            x.plot(df_day.index, df_day[('cm20', k)], linewidth=0.7, color='red', alpha=0.7, linestyle='--')
            x.fill_between(df_day.index, df_day[('low', k)], df_day[('high', k)], color='gray', zorder=-15,
                           alpha=0.5)

            x2 = x.twinx()
            x2.plot(df_day.index, df_day[('ccb', k)], label='ccb', linewidth=0.5, color='blue', marker='.',
                    markersize=4, alpha=0.6)
            x2.plot(df_day.index, df_day[('ccbm20', k)], linewidth=0.5, color='blue', linestyle='-.')
            x2.plot(df_day.index, df_day[('ccbm20', k)], linewidth=0.5, color='blue', linestyle='--')
            x2.fill_between(df_day.index, df_day[('ccbl', k)], df_day[('ccbh', k)], color='blue', zorder=-26,
                            alpha=0.2)
            x2.set_yticks([])

            x.minorticks_on()
            x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
            x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.1)

            x.set_xticks(range(len(df_day))[::tickGapday])
            x.set_xticklabels(xlablesday)



        # 5分钟画图
        for x,k in zip(ax[:,1:].tolist(),  etf_dict.keys()):
            x  = x[0]
            x.plot(df_5min.index, df_5min[('close', k)], label=k, linewidth=1, color='red', marker='.', markersize=4,
                   alpha=0.7)
            x.plot(df_5min.index, df_5min[('cm5', k)], linewidth=0.7, color='red', alpha=0.7, linestyle='-.')
            x.plot(df_5min.index, df_5min[('cm20', k)], linewidth=0.7, color='red', alpha=0.7, linestyle='--')
            x.fill_between(df_5min.index, df_5min[('low', k)], df_5min[('high', k)], color='gray', zorder=-15,
                           alpha=0.5)

            x2 = x.twinx()
            x2.plot(df_5min.index, df_5min[('ccb', k)], label='ccb', linewidth=0.5, color='blue', marker='.',
                    markersize=4, alpha=0.6)
            x2.plot(df_5min.index, df_5min[('ccbma5', k)], linewidth=0.5, color='blue', linestyle='-.')
            x2.plot(df_5min.index, df_5min[('ccbma20', k)], linewidth=0.5, color='blue', linestyle='--')
            x2.fill_between(df_5min.index, df_5min[('ccbl', k)], df_5min[('ccbh', k)], color='blue', zorder=-26,
                            alpha=0.2)
            x2.set_yticks([])

            x.minorticks_on()
            x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
            x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.1)

            x.set_xticks(range(len(df_5min))[::tickGap5min])
            x.set_xticklabels(xlables5min)

        ax[0,0].text(0.1, 1.0, '日K线', horizontalalignment='center', transform=ax[0,0].transAxes, fontsize=12, fontweight='bold', color='black')
        ax[0,1].text(0.5, 1.0, '5分钟K线', horizontalalignment='center', transform=ax[0,1].transAxes, fontsize=12, fontweight='bold', color='black')

        plt.tight_layout()
        plt.suptitle(lastBar, x=0.5, y=0.98)
        plt.savefig('output\\持续监控ETF日Km5K_v2.6_' + datetime.datetime.now().strftime('%Y%m%d') + '.png')

        fig.clf()
        plt.close(fig)

    return


def main():

    global factor, dayr1,png_dict, tdxdata

    if (time.strftime("%H%M", time.localtime()) > '0900' and time.strftime("%H%M", time.localtime()) <= '0930'):
        logger.info('waiting market, sleep 40s')
        time.sleep(40)

    try:

        while (time.strftime("%H%M", time.localtime())>='0930' and time.strftime("%H%M", time.localtime())<='1502'):

            if (time.strftime("%H%M", time.localtime())>'1130' and time.strftime("%H%M", time.localtime())<'1300'):
                logger.info('sleep 60s')
                time.sleep(60)
            else:
                try:
                    png_dict = getMyOptions()
                except:
                    png_dict = {}
                drawAllCCBmin1A5B()
                plotAllzjlx()
                time.sleep(20)

        drawAllCCBmin1A5B()
        plotAllzjlx()

        return
    except Exception as e: 
        logger.info('exception msg: '+ str(e))
        logger.info(' *****  exception, recreate tdxdata then restart main ***** ')
        tdxdata.api.close()
        tdxdata.Exapi.close()
        tdxdata = mytdxData()
        time.sleep(5)
        main()


if __name__ == '__main__':

    prog_start = time.time()
    logger.info('-------------------------------------------')
    logger.info('Job start !!! ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))

    cfg_fn = 'monitor_v2.6.cfg'
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
    plot1min = dict(config.items('plotimgs'))['1min']
    plot5min = dict(config.items('plotimgs'))['5min']
    plot1min5min = dict(config.items('plotimgs'))['1min5min']
    plotday = dict(config.items('plotimgs'))['day']

    tdxdata  = mytdxData()

    now = tdxdata.get_kline_data('399001',0,20,8)
    tempdates = tdxdata.get_kline_data('399001',0,20,9)
    opt_fn_last =  opt_path +  '\\沪深期权清单_'+ tempdates['datetime'].values[-2][:10].replace('-','')+'.csv'
    opt_fn =  opt_path +  '\\沪深期权清单_'+ now['datetime'].values[-1][:10].replace('-','')+'.csv'
    logger.info(opt_fn)

    try:
        png_dict = getMyOptions()
    except:
        png_dict = {}

    factor = calAmtFactor(5)
    factor = factor+[1.00]

    # drawAllCCBmin1A5B()
    # plotAllzjlx()

    main()

    tdxdata.api.close()
    tdxdata.Exapi.close()

    time_end = time.time()
    logger.info('-------------------------------------------')
    logger.info(f'Job completed!!!  All time costed: {(time_end - prog_start):.0f}秒')

