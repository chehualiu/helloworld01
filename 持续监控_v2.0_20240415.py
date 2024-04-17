import json, datetime, os,re
import numpy as np
import warnings
import time, requests
import pandas as pd
import winsound
from playsound import playsound
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import deque
from mplfinance.utils import *

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
import utils.tdxExhq_config as conf
from pytdx.params import TDXParams
from pytdx.reader import TdxDailyBarReader,TdxExHqDailyBarReader
from utils.MyTT import *
from scipy.stats import linregress
from scipy.signal import argrelextrema

# 增加上涨家数 880005的分钟线到 ETF300子图。
# 删除 getRTtickByCode

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# pd.describe_option()
pd.options.mode.chained_assignment = None
# pd.set_option('precision', 6)
pd.set_option('display.precision', 4)
pd.set_option('display.float_format',  '{:.2f}'.format)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('expand_frame_repr', False)
warnings.filterwarnings('ignore')
np.random.seed(42)


import logging
# 创建logger对象
logger = logging.getLogger('mylogger')
# 设置日志等级
logger.setLevel(logging.DEBUG)
# 追加写入文件a ，设置utf-8编码防止中文写入乱码
test_log = logging.FileHandler('logs\\持续监控_v1.9_'+datetime.datetime.now().strftime('%Y%m%d')+ \
                               '.log','a',encoding='utf-8')
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

def TestConnection(Api, type, ip, port):
    if type == 'HQ':
        try:
            is_connect = Api.connect(ip, port)
        except Exception as e:
            logger.error('Failed to connect to HQ!')
            exit(0)

        if is_connect is False:  # 失败了返回False，成功了返回地址
            logger.error('HQ is_connect is False!')
            return False
        else:
            logger.info('HQ is connected!')
            return True

    elif type=='ExHQ':
        try:
            is_connect = Api.connect(ip, port)
        except Exception as e:
            logger.error('Failed to connect to Ext HQ!')
            exit(0)

        if is_connect is False:  # 失败了返回False，成功了返回地址
            logger.error('ExHQ is_connect is False!')
            return False
        else:
            logger.info('ExHQ is connected')
            return True

class tdxData(object):

    def __init__(self, api, Exapi,code, backset, klines, period):

        self.code = code  # 指数名称
        self.backset = backset  # 起始日期
        self.klines = klines  # 结束日期
        self.period = period  # 为指标计算周期
        self.api = api
        self.Exapi = Exapi

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

    def fuquan20231123(self, code, backset, qty, period):

        fuquan = True
        zhishu = False
        if period!=9:           # 如果不是日线级别，跳过复权直接返回。
            fuquan = False
        if code[:2] in ['88','11','12','39','99','zz','zs']:  # 指数不复权
            fuquan = False

        if '#' in code:
            mkt = int(code.split('#')[0])
            code = code.split('#')[1]
            if mkt in [0,1,2] and len(code)!=6: # 深交所0 上交所1 北交所2
                print(code, 'unknown code')
                return pd.DataFrame()

        elif len(code)==6 and code[0] in '0123456789':  # A股
            if code[:2] in ['15','00','30','16','12','39','18']: # 深市
                mkt = 0 # 深交所
            elif code[:2] in ['51','58','56','60','68','50','88','11','99']:
                mkt = 1 # 上交所
            elif code[:2] in ['43','83','87']:
                mkt = 2 # 北交所
            else:
                print(code, 'unknown code')
                return pd.DataFrame()

        elif code[:2].lower()=='zz':    # 中证指数
            code = code[-6:]
            mkt=62
            fuquan = False

        elif code[:2].lower()=='zs':
            fuquan = False
            zhishu = True
            code = code[-6:]
            if code[:2] in ['39']: # 深市
                mkt = 0 # 深交所
            elif code[:2] in ['00']:
                mkt = 1 # 上交所
            elif code[:2] in ['43','83','87']:
                mkt = 2 # 北交所
            else:
                print(code, 'unknown code')
                return pd.DataFrame()

        elif len(code) == 5 and code[0]=='U':    # 期权指标
            mkt = 68

        elif len(code) == 5 and code[0]=='0':    # 港股通
            mkt = 71

        else:
            print(code, 'unknown code')
            return pd.DataFrame()

        if mkt not in [0,1,2]:

            if qty>600:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp,df_k ])
                temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            else:
                df_k = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 0+backset, qty))

            if code[0]=='U':
                return df_k # 扩展接口不复权 直接返回数据
            else:
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

        else:

            if qty>600:
                df_k = pd.DataFrame()
                if code[:2] in ['88','99','39'] or zhishu==True:
                    for i in range(qty//600):
                        temp = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 600*i+backset, 600))
                        df_k = pd.concat([temp,df_k ])
                    temp = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                    df_k = pd.concat([temp, df_k])
                    return df_k # 指数不复权 直接返回数据
                else:
                    for i in range(qty//600):
                        temp = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 600*i+backset, 600))
                        df_k = pd.concat([temp, df_k])
                    temp = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                    df_k = pd.concat([temp, df_k])
            else:
                if code[:2] in ['88','99','39'] or zhishu==True:
                    df_k = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 0+backset, qty))
                    return df_k # 指数不复权 直接返回数据
                else:
                    df_k = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 0+backset, qty))

        if fuquan==False:
            return df_k

        # A股复权
        df_fuquan = self.api.get_xdxr_info(mkt, code)
        if df_fuquan is None:
            print('fuquan code no data', code)
            return df_k
        elif len(df_fuquan) == 0:
            return df_k
        else:
            df_fuquan = pd.DataFrame(df_fuquan, index=[i for i in range(len(df_fuquan))])
            df_fuquan['date'] = df_fuquan.apply(lambda x: str(x.year)+'-'+str(x.month).zfill(2)+'-'+str(x.day).zfill(2), axis=1)

            for i,row in df_fuquan.iterrows():
                if row['date'] not in list(df_k['date']):
                    continue
                elif row['name'] == '除权除息':
                    preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
                    thisclose = df_k.loc[df_k['date']==row['date'], 'close']
                    if row['fenhong']>0 and row['songzhuangu']>0:
                        change_new = (thisclose*(row['songzhuangu']/10+1)+row['fenhong']/10*0.95)/preclose-1
                        df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                    elif row['fenhong']>0:
                        change_new = (thisclose+row['fenhong']/10*0.95)/preclose-1
                        df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                    elif row['songzhuangu']>0:
                        change_new = (thisclose*(row['songzhuangu']/10+1))/preclose-1
                        df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                elif row['name'] == '扩缩股':
                    preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
                    thisclose = df_k.loc[df_k['date']==row['date'], 'close']
                    change_new = (thisclose * row['suogu'])/preclose-1
                    df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                elif row['name'] in ['股本变化','非流通股上市','转配股上市','送配股上市']:
                    continue
                else:
                    print(code, 'unknown name:', row['name'])
            df_k[['open', 'close', 'high', 'low']] = self.cal_right_price(df_k, type='前复权')
            return df_k

        #
        #
        # # 计算牛熊指标
        # def _Calc_func(self, x_df):
        #     '''
        #     输入为df index-date column-turnovet_rate_f|close|pct_chg
        #     ============
        #     return series/pd index-date valus-牛熊指标 kernel_n
        #     '''
        #     df = x_df.copy()
        #     periods = self.periods  # 获取计算周期
        #     if not isinstance(periods, list):
        #         periods = [periods]
        #
        #     # 计算牛熊指标
        #     series_temp = []
        #     for n in periods:
        #         turnover_ma = df['turnover_rate_f'].rolling(n).mean()
        #         std = df['pct_chg'].rolling(n).std(ddof=0)
        #         kernel_factor = std/turnover_ma
        #         kernel_factor.name = 'kernel_'+str(n)
        #         series_temp.append(kernel_factor)
        #
        #     if len(series_temp) == 1:
        #         return kernel_factor
        #
        #     else:
        #         return pd.concat(series_temp, axis=1)
        #
        # # 获取信号
        # @property
        # def get_singal(self):
        #     '''  return df index-date columns-close|pct_chg|kernel_n|turnover_rate_f    '''
        #     data = self.get_data  # 获取数据
        #     singal_df = self._Calc_func(data)  # 计算数据
        #     data = data.join(singal_df)
        #
        #     return data
        #
        # # ---------------------------------------------------------------------
        # #                   信号指标未来收益相关分析函数
        # # ---------------------------------------------------------------------
        #
        # # 计算未来收益
        # def _Cal_forward_ret(self, x_df, singal_periods):
        #     '''
        #     df为信号数据 index-date columns-close|pct_chg|kernel_n
        #     singal_periods int
        #     forwart_n 需要获取未来n日收益的list
        #     ===========
        #     return df
        #
        #     index-date colums kernel_n|未来x日收益|pct_chg
        #     '''
        #
        #     forward_n = self.forward_n  # 获取N日未来收益
        #
        #     df = x_df[['close', 'pct_chg', 'kernel_'+str(singal_periods)]].copy()
        #
        #     # 获取指数数据
        #     df['未来1日收益率'] = df['pct_chg'].shift(-1)  # next_ret
        #
        #     for n in forward_n:
        #         col = '未来{}日收益率'.format(n)
        #         df[col] = df['close'].pct_change(n).shift(-n)  # next_ret_n
        #
        #     df = df.drop('close', axis=1)  # 丢弃不需要的列
        #
        #     return df
        #
        # # 信号分段对应的收益率
        #
        # def forward_distribution_plot(self, singal_periods, bins=50):
        #     '''
        #     bins 分组
        #     =============
        #     return 图表
        #     '''
        #
        #     data = self.get_singal  # 获取信号数据
        #     forward_df = self._Cal_forward_ret(data, singal_periods)  # 获取未来收益
        #     slice_arr = pd.cut(
        #         forward_df['kernel_'+str(singal_periods)], bins=bins)  # 信号分为组
        #
        #     forward_df['barrel'] = slice_arr
        #
        #     group_ret = [x for x in forward_df.columns if x.find(
        #         '日收益率') != -1]  # 获取未来收益列
        #
        #     list_size = len(group_ret)  # 子图个数
        #
        #     # 画图
        #     f, axarr = plt.subplots(list_size, figsize=(18, 15))  # sharex=True 画子图
        #     f.suptitle(' 信号对应收益分布')  # 设置总标题
        #
        #     # 设置子图间隔等
        #     f.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.65,
        #                       wspace=0.35)
        #
        #     for i in range(list_size):
        #
        #         ret_series = forward_df.groupby('barrel')[group_ret[i]].mean()
        #         x = ret_series.index.astype('str').tolist()
        #
        #         axarr[i].bar(x, ret_series.values)  # 画子图
        #         axarr[i].set_title(group_ret[i])  # 设置子图名称
        #         axarr[i].set_xticklabels(x, rotation=90)  # 设置子图刻度标签及标签旋转
        #
        #     plt.show()
        #
        # # 画分信号位数与收益的相关系数
        # def QuantReg_plot(self, forward_ret_name, singal_periods):
        #     '''
        #     forward_ret_name 需要比较的未来收益的名称
        #     '''
        #     data = self.get_singal  # 获取信号数据
        #
        #     forward_df = self._Cal_forward_ret(data, singal_periods)  # 获取未来收益
        #
        #     x = forward_ret_name  # 设置未来收益名称
        #     traget = 'kernel_'+str(singal_periods)+'~'+x
        #
        #     mod = smf.quantreg(traget, forward_df)
        #     res = mod.fit(q=.5)
        #     print(res.summary())
        #
        #     quantiles = np.arange(.05, .96, .1)
        #
        #     def fit_model(q):
        #         res = mod.fit(q=q)
        #         # conf_int为置信区间
        #         return [q, res.params['Intercept'], res.params[x]] + \
        #                res.conf_int().loc[x].tolist()
        #
        #     models = [fit_model(q_i) for q_i in quantiles]
        #     models_df = pd.DataFrame(
        #         models, columns=['quantiles', 'Intercept', x, 'lb', 'ub'])
        #
        #     n = models_df.shape[0]
        #
        #     plt.figure(figsize=(18, 8))
        #     p1 = plt.plot(models_df['quantiles'], models_df[x],
        #                   color='red', label='Quantile Reg.')
        #     p2 = plt.plot(models_df['quantiles'], models_df['ub'],
        #                   linestyle='dotted', color='black')
        #     p3 = plt.plot(models_df['quantiles'], models_df['lb'],
        #                   linestyle='dotted', color='black')
        #
        #     plt.ylabel(r'$\beta_{RETURNS}$')
        #     plt.xlabel('信号分位数')
        #     plt.legend()
        #     plt.show()

    @property
    def get_data(self):
        df=self.fuquan20231123(self.code, self.backset, self.klines, self.period)
        # df['trade_date']=df['date']
        # df['pctChg']=df['close']/df['close'].shift(1)-1
        return df


def getOptionsTformat(df_4T):

    field_map3 = {'f14':'Cname','f12':'Ccode','f2':'Cprice', 'f3':'CpctChg','f4':'C涨跌额','f108':'C持仓量','f5':'Cvol','f249':'Civ','f250':'C折溢价率','f161':'行权价',
                  'f340':'Pname','f339':'Pcode','f341':'Pprice','f343':'PpctChg','f342':'P涨跌额','f345':'P持仓量','f344':'Pvol','f346':'Piv','f347':'P折溢价率'}

    df_T_data = pd.DataFrame()
    for etfcode, expiredate in zip(df_4T['ETFcode'],df_4T['到期日']):
        code= '1.'+etfcode if etfcode[0]=='5' else '0.'+etfcode
        # 89.push2   >>  push2
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

    # url4 = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112308418460865815227_1695516975860&fid=f3&po=1&'+\
    #        'pz='+'50'+'&pn='+'1'+'&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5'+\
    #        '&fields=f1,f2,f3,f12,f13,f14,f302,f303,f325,f326,f327,f329,f328,f301,f152,f154&fs=m:10'

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
    ############################### base info
    field_map0 = {'f12':'code', 'f14':'name','f301':'到期日','f331':'ETFcode', 'f333':'ETFname',
                  'f2':'close','f3':'pctChg','f334':'ETFprice','f335':'ETFpct','f337':'平衡价',
                  'f250':'溢价率','f161':'行权价'}#,'f47':'vol','f48':'amount','f133':'allvol'}

    # Shanghai options
    url1 = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112307429657982724098_1687701611430&fid=f250'+ \
           '&po=1&pz=1000&pn=1&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5'+ \
           '&fields=f1,f2,f3,f12,f13,f14,f161,f250,f330,f331,f332,f333,f334,f335,f337,f301,f152&fs=m:10'
    res = requests.get(url1, headers=header)
    tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
    data1 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    # data1 = data1[['f12', 'f14']]
    data1.rename(columns = field_map0,inplace=True)

    # Shenzhen options
    url2 = url1[:-1] + '2'
    res = requests.get(url2,headers=header)
    tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
    data2 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    # data2= data2[['f12', 'f14']]
    data2.rename(columns = field_map0,inplace=True)
    # data2['交割价'] = data2['name'].apply(lambda x: float(x[-5:-1])/1000 if x[-1]=='A' else float(x[-4:])/1000)

    data = pd.concat([data1, data2])
    data = data[list(field_map0.values())]

    data['market'] = data['ETFcode'].apply(lambda x: '沪市' if x[0]=='5' else '深市')
    data['direction'] = data['name'].apply(lambda x: 'call' if '购' in x else 'put')
    data['due_date'] = data['到期日'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d').date())
    data['dte'] = data['due_date'].apply(lambda x: (x-datetime.datetime.now().date()).days)
    data['close'] = data['close'].astype(float)
    data['到期日'] = data['到期日'].astype(str)
    data['行权pct'] = data.apply(lambda x:round(x['行权价']/x['ETFprice']*100-100,2),axis=1)
    # del data['f1'],	data['f13'], data['f152'], data['f330'], data['f332']

    df_4T = data.pivot_table(index=['ETFcode','到期日'],values=['name'],aggfunc=['count']).reset_index()
    df_4T.columns = ['ETFcode','到期日','数量']


    ######################## T-型报价
    df_T_format = getOptionsTformat(df_4T)
    tempc = df_T_format[['Ccode','C持仓量','Cvol']]
    tempc.columns = ['code','持仓量','vol']
    tempp = df_T_format[['Pcode','P持仓量','Pvol']]
    tempp.columns = ['code','持仓量','vol']
    temp = pd.concat([tempc, tempp])
    temp['vol'] = temp['vol'].astype(int)
    data = pd.merge(data, temp, on='code',how='left')
    data['amount'] = data['close']*data['vol']

    ######################## 风险分析
    df_risk = getOptionsRiskData()

    data = pd.merge(data, df_risk[['code','杠杆比率','实际杠杆', 'Delta','Gamma','Vega','Theta','Rho']], on='code',how='left')

    return data

def getMyOptions():
    global dte_high, dte_low,close_Threshold

    now = pd.DataFrame(api.get_index_bars(8, 1, '999999', 0, 20))

    opt_fn = r'D:\stockProg\win2023\output\options\沪深期权清单_'+ now['datetime'].values[-1][:10].replace('-','')+'.csv'
    current_datetime = datetime.datetime.strptime(now['datetime'].values[-1],'%Y-%m-%d %H:%M')

    if os.path.exists(opt_fn):
        modified_timestamp = os.path.getmtime(opt_fn)
        modified_datetime = datetime.datetime.fromtimestamp(modified_timestamp)
        time_delta = current_datetime - modified_datetime
        gap_seconds = time_delta.days*24*3600 + time_delta.seconds
        if gap_seconds < 1000:
            print('\nreusing option file', opt_fn)
            data = pd.read_csv(opt_fn, encoding='gbk',dtype={'ETFcode':str,'code':str})
        else:
            try:
                data = getAllOptionsV3()
                print('\nNew option file', opt_fn)
                data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
            except:
                print('\nupdate failed, reusing option file', opt_fn)
                data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    else:
        print('\nNew option file', opt_fn)
        data = getAllOptionsV3()
        data.to_csv(opt_fn,encoding='gbk',index=False, float_format='%.4f')

    data.fillna(0,inplace=True)
    amtlist = data['amount'].values.tolist()
    amtlist.sort()
    amtthreshold = amtlist[-100]

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

def playalarm(sig=1):
    pass
    return
    # if sig==1:
    #     playsound('D:\\mp3\\effect\\morning.mp3')
    # elif sig==-1:
    #     playsound('D:\\mp3\\effect\\swoosh.mp3')
    # else:
    #     print('playalarm no para')

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
        logger.error('load data file failed')
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
    # min['northdeltam5'] = min['northdelta'].rolling(5).mean()

    return min

def getSouthzjlx():

    url = 'https://push2.eastmoney.com/api/qt/kamtbs.rtmin/get?fields1=f1,f2,f3,f4&fields2=f51,f54,f52,f58,f53,f62,f56,f57,f60,f61&'+ \
          'ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery112309012195417245222_1655353679439&_=1655353679440'

    # res = requests.get(url)
    res = requests.get(url)

    try:
        data1 = json.loads(res.text[42:-2])['data']['n2s']
    except:
        return pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1],columns=['time', 'hgtnet', 'hgtin', 'sgtnet', 'hgtout', 'south', 'sgtin','sgtout', 'northin', 'northout'])
    min = min[30:]
    min.reset_index(drop=True, inplace=True)
    min.drop(labels=['time', 'hgtnet', 'hgtin', 'sgtnet', 'hgtout', 'sgtin','sgtout', 'northin', 'northout'],axis=1,inplace=True)
    min = min[min['south']!='-']
    # min['time'] = min['time'].astype('datetime64[ns]')
    min['south'] = min['south'].astype('float')/10000
    min['southdelta'] = min['south'] - min['south'].shift(1)
    # min['southdeltam5'] = min['southdelta'].rolling(5).mean()

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
    # min.rename(columns={'mea'})
    # data['time'] = data['time'].astype('datetime64[ns]')
    data['close'] = data['close'].astype('float')
    data['vol'] = data['vol'].astype('float')
    data['amt'] = data['close']*data['vol']
    data['amtcum'] = data['amt'].cumsum()
    data['volcum'] = data['vol'].cumsum()
    data['avg'] = data['amtcum']/data['volcum']
    data['amttrend'] = 0

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

def MINgetDPindex():
    global factor

    day_sh =  pd.DataFrame(api.get_index_bars(9, 1, '999999', 0, 30))
    datelast = day_sh['datetime'].values[-2]
    preclose = day_sh[day_sh['datetime']==datelast]['close'].values[-1]

    df_sh =  pd.DataFrame(api.get_index_bars(8, 1, '999999', 0, 300))
    df_sh['date'] = df_sh['datetime'].apply(lambda x: x[:10])
    df_sh['time'] = df_sh['datetime'].apply(lambda x: x[11:])
    df_sh = df_sh[(df_sh['datetime']>datelast)]
    df_sh.reset_index(drop=True,inplace=True)

    df_sz =  pd.DataFrame(api.get_index_bars(8, 0, '399001', 0, 300))
    df_sz['date'] = df_sz['datetime'].apply(lambda x: x[:10])
    df_sz['time'] = df_sz['datetime'].apply(lambda x: x[11:])
    df_sz = df_sz[(df_sz['datetime']>datelast)]
    df_sz.reset_index(drop=True,inplace=True)
    df_sz.rename(columns={'amount':'amountsz'}, inplace=True)

    data = pd.merge(df_sh[['time','amount','vol','close']], df_sz[['time','amountsz']], how='inner')
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

    ttt = df_min.pivot_table(index='time',values='amount',aggfunc='sum')
    ttt['cum'] = ttt['amount'].cumsum()
    ttt['ratio'] = ttt['cum'].values[-1] / ttt['cum']
    # amt_all = df_min['amount'].sum()
    # df_result = pd.DataFrame(data=df_min['time'].values[0:240], columns=['minute'])
    # df_result['minpct'] = 0
    # for min in df_min['time'].values[0:240]:
    #     amt_min = df_min[df_min['time']==min]['amount'].sum()
    #     df_result.loc[df_result['minute']==min, 'minpct'] = amt_min / amt_all
    # df_result['pct'] = df_result['minpct'].cumsum()
    # df_result['factor'] = 1/df_result['pct']

    return list(ttt['ratio'].values)

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
        logger.error(str(market)+' '+ code+ " no data.")
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
        logger.error('code not accepted '+  code)

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
            logger.error('sEM_error ' +res.text)
            return []

    except requests.exceptions.RequestException:
        logger.error('sina_error '+code)
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
        logger.error('exception msg: '+ str(e))
        logger.error(' -----  exception, restart main ----- ')
        time.sleep(5)
        main()

def processSingleStock1min(code,name,api,Exapi):
    global dayr1

    try:

        # df_1min = getStk1min(code, name, api,Exapi)
        df_1min = getStk1minV2(code, name, api,Exapi)
        if len(df_1min) == 0:
            logger.error(code+' '+name +' TdxRealtime failed.')
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
                    logger.warning('DOWN ' +code + ' '+ name + ' ' + ' 跌破顶部左侧低点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                          '  止损价:' + str(round(low1,3)) + ' 止损%:' + str(round(low1/df_1min['close'].values[-1]*100-100,2)) + '%')
                    flag = -1
                    # playalarm()
                elif cnow<low2 and cr2>low2 and df_1min['close'].values[-1]<df_1min['ma10'].values[-1]:
                    logger.warning('DOWN ' +code + ' '+ name + ' ' + ' 跌破顶部右侧低点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                                   '  止损价:' + str(round(low2,3)) + ' 止损%:' + str(round(low2/df_1min['close'].values[-1]*100-100,2)) + '%')
                    flag = -1
                    # playalarm()
                else:
                    return
            elif idxnow==high0idx and idxnow-df_1min[:-1]['close'].argmax()>=10 and idxnow-low0idx>25 and cnow>df_1min[:-1]['close'].max(): # 刚创新高
                high1idx = df_1min[:-1]['close'].argmax()
                low1 = df_1min[high1idx:]['close'].min()
                logger.warning('UP '+code + ' '+ name  + ' ' + ' 上升中刚创新高 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                      '  止损价:' + str(round(low1,3)) + ' 止损%:' + str(round(df_1min['close'].values[-1]/low1*100-100,2)) + '%')
                flag = 2
                playalarm(1)
            else:
                return
        else:   #  high0idx<low0idx 下跌中
            if idxnow-low0idx>15: # 绝对低点在15分钟前
                high2 = df_1min[low0idx:]['close'].max()    # 低点右侧高点
                high1 = df_1min[:low0idx][df_1min["close"]==df_1min['max']]['close'].min() # 低点左侧高点

                if cnow>high1 and cr2<high1 and df_1min['close'].values[-1]>df_1min['ma10'].values[-1]:  # cnow>highThreashold and cr2<highThreasholdr2
                    logger.warning('UP ' +code + ' '+ name  + ' ' + ' 突破底部左侧高点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                          '  止损价:' + str(round(high1,3)) + ' 止损%:' + str(round(df_1min['close'].values[-1]/high1*100-100,2)) + '%')
                    flag = 1
                    playalarm(1)
                elif cnow>high2 and cr2<high2 and df_1min['close'].values[-1]>df_1min['ma10'].values[-1]:
                    logger.warning('UP ' +code + ' '+ name  + ' ' + ' 突破底部右侧高点 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                                   '  止损价:' + str(round(high2,3)) + ' 止损%:' + str(round(df_1min['close'].values[-1]/high2*100-100,2)) + '%')
                    flag = 1
                    playalarm(1)
                else:
                    return
            elif idxnow==low0idx and idxnow-df_1min[:-1]['close'].argmin()>=10 and idxnow-high0idx>25 and cnow<df_1min[:-1]['close'].min(): # 刚创新低
                low1idx = df_1min[:-1]['close'].argmin()
                high1 = df_1min[low1idx:]['close'].max()
                logger.warning('DOWN ' +code + ' '+ name  + ' ' + ' 下跌中刚创新低 -- '+  ' close '+  str(round(df_1min['close'].values[-1],3))+ \
                      '  止损价:' + str(round(high1,3)) + ' 止损%:' + str(round(high1/df_1min['close'].values[-1]*100-100,2)) + '%')
                flag = -2
                # playalarm(-1)
            else:
                return

        return

    except:
        logger.error('processing '+ code+' '+ name+ ' error')
        return

def plotAllzjlx():
    df_north = getNorthzjlx()
    df_south = getSouthzjlx()
    df_southNorth = pd.merge(df_north, df_south, left_index=True, right_index=True)
    hs300_zjlx = getHS300zjlx()  # zjlxdelta
    hs300_index,hs_preclose = getETFindex('510300')  #   close
    if len(hs300_index) == 0:
        logger.error('failed to get 300ETF df, skipped')
        return
    dp_zjlx = MINgetZjlxDP()   # zjlxdelta
    dp_index, dp_preclose = MINgetDPindex()  # close

    dp_boss = dp_zjlx.boss.values[-1]/100000000
    north_amt = df_north.north.values[-1]
    south_amt = df_south.south.values[-1]
    hs300_boss = hs300_zjlx.boss.values[-1]
    upcnt = hs300_index.upcnt.values[-2]
    # df_dp = pd.merge(df_north, dp_zjlx, left_index=True, right_index=True)
    df_dp = pd.merge(df_southNorth, dp_zjlx, left_index=True, right_index=True)
    df_dp = pd.merge(df_dp, dp_index, left_index=True, right_index=True)
    df_hs300 = pd.merge(hs300_zjlx, hs300_index,left_index=True, right_index=True)

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
        logger.warning('UP 510300 hs300' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = 'http://wx.xtuis.cn/XfhmghWTRzitW6RHalZc8AzN5.send?text=' + msg
        requests.get(msgURL)
        msgURL = 'http://wx.xtuis.cn/bWwwuJjL3EojjNJDwool6kfeI.send?text=' + msg
        requests.get(msgURL)
        playalarm(1)
    if df_hs300['dw'].values[-1]==True:
        msg = 'UP 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))
        logger.warning('DOWN 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = 'http://wx.xtuis.cn/XfhmghWTRzitW6RHalZc8AzN5.send?text=' + msg
        requests.get(msgURL)
        msgURL = 'http://wx.xtuis.cn/bWwwuJjL3EojjNJDwool6kfeI.send?text=' + msg
        requests.get(msgURL)
        playalarm(-1)
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
        logger.warning('UP 999999 沪市大盘' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_dp['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        playalarm(1)
    if df_dp['dw'].values[-1]==True:
        logger.warning('DOWN 999999 沪市大盘' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_dp['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        playalarm(-1)
    df_dp['up'] = df_dp.apply(lambda x: x.close if x.up==True else np.nan, axis=1)
    df_dp['dw'] = df_dp.apply(lambda x: x.close if x.dw==True else np.nan, axis=1)
    if len(df_dp)<240:
        df_dp = pd.concat([df_dp, pd.DataFrame([[]]*(240-len(df_dp)))])
    df_dp.reset_index(drop=True,inplace=True)
    df_dp.reset_index(inplace=True)
    df_dp.iloc[0:2, df_dp.columns.get_loc('amttrend')] = np.nan

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
    ax2d = ax2.twinx()

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
    # ax1c.plot(df_dp.index, df_dp.northdeltam5, label='northdeltam5', color='black', lw=0.4)
    ax1c.hlines(y=0, xmin=0, xmax=240-3, colors='black', linestyles='-', lw=0.3)
    ax1d.plot(df_dp.index, df_dp.amttrend, label='amttrend', color='green', lw=1.5, alpha=0.5)
    ax1.text(0.5,0.92,' 大盘主力资金(蓝柱):' + str(round(dp_boss,2)) + ' 北向流入(灰柱):' + str(round(north_amt,2)) + ' 量能(绿线):'+ dp_amount,
             horizontalalignment='center',transform=ax1.transAxes, fontsize=12, fontweight='bold', color='black')

    df_hs300.plot(x='index', y='close', label='hs300', linewidth=1, color='red', ax=ax2,zorder=10)
    df_hs300.plot(x='index', y='avg', linewidth=1, markersize=10, color='violet', label='avg', ax=ax2,zorder=11)
    ax2.hlines(y=hs_preclose, xmin=0, xmax=240-3, colors='aqua', linestyles='-', lw=2, label='preclose')
    ax2.scatter(df_hs300.index, df_hs300['up'], marker='^', s=100, c='red',alpha=0.7)
    ax2.scatter(df_hs300.index, df_hs300['dw'], marker='v',s=100, c='green',alpha=0.7)
    ax2b.bar(df_hs300.index, df_hs300.net, label='zjlx', color='blue', alpha=0.2, zorder=-15)
    ax2b.plot(df_hs300.index, df_hs300.bossma5, label='zjlxm5', color='blue', lw=0.5)
    ax2c.plot(df_hs300.index, df_hs300.upcnt, label='upcnt', color='green', lw=1.5,alpha=0.5 )
    ax2d.bar(df_dp.index, df_dp.southdelta, label=None, color='grey', alpha=0.5, zorder=-14)
    ax2d.hlines(y=0, xmin=0, xmax=240-3, colors='black', linestyles='-', lw=0.3)
    ax2.text(0.5,0.9,'HS300 主力流入(蓝柱):' + str(round(hs300_boss,2))+' 南向流入(灰柱):' + str(round(south_amt,2)) + ' 上涨数(绿线): '+ str(round(upcnt,0)),
             horizontalalignment='center',transform=ax2.transAxes, fontsize=12, fontweight='bold', color='black')
    ax1b.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax1c.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax2b.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax2d.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)

    # lines = ax.lines[3:] + ax3.lines[1:]
    # ax.get_legend().remove()
    # labs = [l.get_label() for l in lines]
    # ax.legend(lines, labs, loc=0)

    func1 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x,(x/dp_preclose-1)*100)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(func1))
    func2 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x,(x/hs_preclose-1)*100)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(func2))

    ax1.minorticks_on()
    ax1.grid(which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    # ax1.grid(which='minor', axis="both", color='k', linestyle='dotted', linewidth=0.5)
    ax1.set(xlabel=None)

    ax2.minorticks_on()
    ax2.grid(which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    # ax2.grid(which='minor', axis="both", color='k', linestyle='dotted', linewidth=0.5)
    ax2.set(xlabel=None)

    plt.suptitle('DP HS300 - 时间戳 ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
    plt.tight_layout()

    plt.savefig('output\\持续监控DP_HS300_'+ datetime.datetime.now().strftime('%Y%m%d') + '.png' )

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


def fuquan20231123(api, code, backset, qty, period):

    if '#' in code:
        mkt = int(code.split('#')[0])
        code = code.split('#')[1]
        if qty>600:
            df_k = pd.DataFrame()
            for i in range(qty//600):
                temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                df_k = pd.concat([temp,df_k ])
            temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
            df_k = pd.concat([temp, df_k])
        else:
            df_k = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 0+backset, qty))
        return df_k

    elif len(code)==6:
        if code[:2] in ['15','00','30','16','12','39','18']: # 深市
            mkt = 0 # 深交所
        elif code[:2] in ['51','58','56','60','68','50','88','11','99']:
            mkt = 1 # 上交所
        elif code[:2] in ['43','83','87']:
            mkt = 2 # 北交所
        else:
            print(code, 'unknown code')
            return pd.DataFrame()

        if qty>600:
            df_k = pd.DataFrame()
            if code[:2] in ['88','99','39']:
                for i in range(qty//600):
                    temp = pd.DataFrame(api.get_index_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp,df_k ])
                temp = pd.DataFrame(api.get_index_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            else:
                for i in range(qty//600):
                    temp = pd.DataFrame(api.get_security_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp, df_k])
                temp = pd.DataFrame(api.get_security_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
        else:
            if code[:2] in ['88','99','39']:
                df_k = pd.DataFrame(api.get_index_bars(period, mkt, code, 0+backset, qty))
            else:
                df_k = pd.DataFrame(api.get_security_bars(period, mkt, code, 0+backset, qty))
    elif code[:2].lower()=='zz':
        # code = code[-6:]
        mkt=62
        if qty>600:
            df_k = pd.DataFrame()
            for i in range(qty//600):
                temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code[-6:], 600*i+backset, 600))
                df_k = pd.concat([temp, df_k])
            temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code[-6:], 600*(qty//600)+backset, qty%600))
            df_k = pd.concat([temp, df_k])
        else:
            df_k = pd.DataFrame(api.get_instrument_bars(period, mkt, code[-6:], 0+backset, qty))
    elif code[:2].lower()=='zs':
        code = code[-6:]
        if code[:2] in ['39']: # 深市
            mkt = 0 # 深交所
        elif code[:2] in ['00']:
            mkt = 1 # 上交所
        elif code[:2] in ['43','83','87']:
            mkt = 2 # 北交所
        else:
            print(code, 'unknown code')
            return pd.DataFrame()
        if qty>600:
            df_k = pd.DataFrame()
            for i in range(qty//600):
                temp = pd.DataFrame(api.get_index_bars(period, mkt, code, 600*i+backset, 600))
                df_k = pd.concat([temp, df_k])
            temp = pd.DataFrame(api.get_index_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
            df_k = pd.concat([temp, df_k])
            return df_k
        else:
            df_k = pd.DataFrame(api.get_index_bars(period, mkt, code, 0+backset, qty))
            return df_k
    elif len(code) == 5:
        if code[0]=='U':    # 期权指标
            mkt = 68
            if qty>600:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp, df_k])
                temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            else:
                df_k = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 0+backset, qty))
            return df_k
        else:
            mkt = 71    # 港股通
            if qty>600:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp, df_k])
                temp = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            else:
                df_k = pd.DataFrame(api.get_instrument_bars(period, mkt, code, 0+backset, qty))
    else:
        print(code, 'unknown code')
        return pd.DataFrame()

    # 如果不是日线级别，跳过复权直接返回。
    if period!=9:
        return df_k

    if len(df_k)<10:
        print(code, 'pytdx no data')
        return pd.DataFrame()
    df_k['date'] = df_k['datetime'].apply(lambda x: x[:10])
    df_k['preclose'] = df_k['close'].shift(1)
    df_k['change'] = df_k['close']/df_k['close'].shift(1)-1
    df_k.dropna(subset=['preclose'], inplace=True)


    # 港股复权
    if len(code)==5:
        df_fuquan = get_xdxr_EM(code)
        if len(df_fuquan)==0:
            return df_k
        for i,row in df_fuquan.iterrows():
            if row['date'] not in list(df_k['date']):
                continue
            preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
            thisclose = df_k.loc[df_k['date']==row['date'], 'close']
            change_new = (thisclose+row['deal']*0.99)/preclose-1
            df_k.loc[df_k['date']==row['date'], 'change'] = change_new
        df_k[['open', 'close', 'high', 'low']] = cal_right_price(df_k, type='前复权')
        return df_k

    # A股复权 - 略过
    if code[:2] in ['88','11','12','39','99','zz']:  # 指数不复权
        return df_k

    # A股复权
    df_fuquan = api.get_xdxr_info(mkt, code)
    if df_fuquan is None:
        print('fuquan code no data', code)
        return df_k
    elif len(df_fuquan) == 0:
        return df_k

    df_fuquan = pd.DataFrame(df_fuquan, index=[i for i in range(len(df_fuquan))])
    df_fuquan['date'] = df_fuquan.apply(lambda x: str(x.year)+'-'+str(x.month).zfill(2)+'-'+str(x.day).zfill(2), axis=1)

    for i,row in df_fuquan.iterrows():
        if row['date'] not in list(df_k['date']):
            continue
        elif row['name'] == '除权除息':
            preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
            thisclose = df_k.loc[df_k['date']==row['date'], 'close']
            if row['fenhong']>0 and row['songzhuangu']>0:
                change_new = (thisclose*(row['songzhuangu']/10+1)+row['fenhong']/10*0.95)/preclose-1
                df_k.loc[df_k['date']==row['date'], 'change'] = change_new
            elif row['fenhong']>0:
                change_new = (thisclose+row['fenhong']/10*0.95)/preclose-1
                df_k.loc[df_k['date']==row['date'], 'change'] = change_new
            elif row['songzhuangu']>0:
                change_new = (thisclose*(row['songzhuangu']/10+1))/preclose-1
                df_k.loc[df_k['date']==row['date'], 'change'] = change_new
        elif row['name'] == '扩缩股':
            preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
            thisclose = df_k.loc[df_k['date']==row['date'], 'close']
            change_new = (thisclose * row['suogu'])/preclose-1
            df_k.loc[df_k['date']==row['date'], 'change'] = change_new
        elif row['name'] in ['股本变化','非流通股上市','转配股上市','送配股上市']:
            continue
        else:
            print(code, 'unknown name:', row['name'])

    df_k[['open', 'close', 'high', 'low']] = cal_right_price(df_k, type='前复权')
    return df_k


def getSingleCCBData(name, period, backset, klines,mapara):
    global api, Exapi
    if period==0:
        code = etf_dict2[name]
    else:
        code = etf_dict[name]
    df_single= tdxData(api, Exapi,code,backset,klines,period).get_data
    df_single.reset_index(drop=True,inplace=True)
    if len(df_single)==0:
        print(code,'kline error,quitting')
        return
    df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30') if x[-5:]=='13:00' else x)
    # df_single['date'] = pd.to_datetime(df_single['datetime'])
    # lista = getRenko(df_single[['date','datetime','open','high','low','close']])
    # df_single['cRenkoM10'] = lista

    ccbcode = etf_ccb_dict[name]
    df_ccb = tdxData(api, Exapi, ccbcode,backset,klines,period).get_data
    if len(df_ccb)==0:
        print(code,'ccb error, quitting')
        return
    # tmp = df_ccb[['datetime','open','high','low','close']]
    # tmp['date'] = pd.to_datetime(tmp['datetime'])
    # lista = getRenko(tmp)
    # df_ccb['ccbRenkoM10'] = lista
    df_ccb.rename(columns={'close':'ccb','high':'ccbh','low':'ccbl','open':'ccbo'},inplace=True)
    data = pd.merge(df_ccb[['datetime','ccb','ccbh','ccbl','ccbo']], df_single[['datetime','open','close','high','low']], on='datetime',how='left')

    return data

def get_threshold(ser,mybins,pct):

    data_values = ser.values
    bins = np.histogram_bin_edges(data_values, bins=mybins, range=(min(data_values), max(data_values)))
    hist, _ = np.histogram(data_values, bins=bins)
    cumulative_hist = np.cumsum(hist)
    idx = list(cumulative_hist).index(cumulative_hist[cumulative_hist>pct/100*len(data_values)][0])
    return bins[idx]

def plot_hist(ser,bins,pct):

    data_values = ser.values
    # Make sure 'data_values' is a 1D array before calling hist
    data_values = data_values.reshape(-1)
    n, bins, patches = plt.hist(data_values, bins=bins, cumulative=1, density=True, histtype='step', color='red')#, return_values=True)
    plt.close()
    idx = list(n).index(n[n>pct/100][0])
    return bins[idx]

def CROSS(S1, S2):  # 判断向上金叉穿越 CROSS(MA(C,5),MA(C,10))  判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # 不使用0级函数,移植方便  by jqz1226

def getRenko(data):
    df = data.copy()
    marketcolors = {'candle': {'up': 'w', 'down': 'k'},
                     'edge': {'up': 'k', 'down': 'k'},
                     'wick': {'up': 'k', 'down': 'k'},
                     'ohlc': {'up': 'k', 'down': 'k'},
                     'volume': {'up': '#1f77b4',
                                'down': '#1f77b4'},
                     'vcedge': {'up': '#1f77b4',
                                'down': '#1f77b4'},
                     'vcdopcod': False, 'alpha': 0.9}

    dates, opens, highs, lows, closes = df['date'].values, df['open'].values, df['high'].values, df['low'].values, df['close'].values
    collections = construct_renko_collections(dates, highs, lows, None, {}, closes, marketcolors=marketcolors)


    temp = pd.DataFrame(data={'date': collections[1]['dates'], 'close': collections[1]['values']})
    temp['renko_ma10'] = temp['close'].rolling(20).mean()
    temp['time'] = [i[:-3].replace('T',' ') for i in temp['date'].astype(str)]
    temp = temp.drop_duplicates(subset=['time'])
    # temp = pd.merge(temp, df[['datetime','high','low']],left_on='time',right_on='datetime',how='left')
    # temp['vally'] = (temp['close'] == temp['close'].rolling(3).min()) & (temp['close'] < temp['close'].shift(-1)) &  (temp['close'] < temp['close'].shift(-2))
    # temp['peak'] = (temp['close'] == temp['close'].rolling(3).max()) & (temp['close'] > temp['close'].shift(-1)) & (temp['close'] > temp['close'].shift(-2))
    # temp.loc[temp['vally']==True, 'point'] = temp['low']
    # temp.loc[temp['peak']==True, 'point'] = temp['high']
    # df = pd.merge(df,temp[['time','point','renko_ma10']],left_on='datetime',right_on='time',how='left')

    df = pd.merge(df,temp[['time','renko_ma10']],left_on='datetime',right_on='time',how='left')
    # df_single['point'] = df_single['point'].ffill()
    df['renko_ma10'] = df['renko_ma10'].ffill()
    # df.iloc[0, df.columns.get_loc('point')] = df['close'].values[0]
    # df.iloc[-1, df.columns.get_loc('point')] = df['close'].values[-1]
    # df.reset_index(inplace=True)

    # df_renko = df_single[df_single['point'].notnull()]
    return df['renko_ma10'].to_list()

def absAccum(l):
    acc = []
    for num in l:
        if len(acc)>0 and acc[-1]<0 and num > 0:
            acc.append(num)
        elif len(acc)>0 and acc[-1]>0 and num > 0:
            acc.append(acc[-1] + num)
        elif len(acc)>0 and acc[-1]>0 and num < 0:
            acc.append(num)
        elif len(acc)>0 and acc[-1]<0 and num < 0:
            acc.append(acc[-1] + num)
        else:
            acc.append(num)
    return(acc)

def barsAccum(l):
    acc = []
    # ttl = 0
    for num in l:
        if len(acc)>0 and acc[-1]<0 and num > 0:
            acc.append(num)
            # ttl = 1
        elif len(acc)>0 and acc[-1]>0 and num > 0:
            acc.append(acc[-1] + num)
        elif len(acc)>0 and acc[-1]>0 and num < 0:
            acc.append(num)
        elif len(acc)>0 and acc[-1]<0 and num < 0:
            acc.append(acc[-1] + num)
        else:
            acc.append(num)
            # ttl = 1
    return(acc)

def calcBS(data):

    df = data.copy()
    C=df['close']; H=df['high']; L=df['low']; CCB=df['ccb']; CCBH=df['ccbh']; CCBL=df['ccbl']
    CM20=df['cm20']; CM5=df['cm5']; CCBMA5=df['ccbma5']; CCBMA20=df['ccbma20']
    CM20ABS = ABS(CM20/REF(CM20,1)-1)

    # S1 - M20均线下行, ref(c,1)>ma20， C<ref(C,1)
    S1 = (COUNT(CM20<REF(CM20,1),20)>15) & (REF(C,1)>REF(CM20)) & (C<REF(C,1))
    # B1 - M20均线上行, ref(c,1)<ma20， C>ref(C,1)
    B1 = (COUNT(CM20>REF(CM20,1),20)>15) & (REF(C,1)>REF(CM20)) & (C<REF(C,1))
    B = B1
    S = S1

    # B = plot_hist(df_single["gapabs"], bins, gaphist_pct)

    # df.loc[B,'B']=True
    # df.loc[S,'S']=True

    return B.values.tolist(),S.values.tolist()


    # df['pctR10'] = df['close'] / df['close'].shift(10) - 1
    # df['pctR10MA5'] = df['pctR10'].rolling(5).mean()
    #
    # df['dd'] = df['low'] < df['low'].shift(1)
    # df['gd'] = df['high'] > df['high'].shift(1)
    # df['pctR10R1'] = df['pctR10'].shift(1)
    # df['pctR10LLV20'] = df['pctR10'].rolling(20).min()
    # df['pctR10HHV20'] = df['pctR10'].rolling(20).max()
    # df['1buy'] = df.apply(lambda x: True if x.dd and x.pctR10<x.pctR10LLV20*0.7 and x.pctR10>x.pctR10R1 else False, axis=1)
    # df['1sell'] = df.apply(lambda x: True if x.gd and x.pctR10>x.pctR10HHV20*0.7 and x.pctR10<x.pctR10R1 else False, axis=1)
    #
    # df['HHV9'] = df['high'].rolling(9).max()
    # df['LLV9'] = df['low'].rolling(9).min()
    # df['RSV'] = df.apply(lambda x: (x.close - x.LLV9) / (x.HHV9 - x.LLV9) * 100, axis=1)
    # df['K'] = talib.SMA(df['RSV'], 3)
    # df['D'] = talib.SMA(df['K'], 3)
    # df['J'] = df.apply(lambda x: x.K * 3 - x.D * 2, axis=1)
    # df['CP20'] = (df['close'] - df['low'].rolling(20).min()) / (
    #             df['high'].rolling(20).max() - df['low'].rolling(20).min())
    #
    # df['up1'] = df['up2'] = df['up3'] = False
    # df['dw1'] = df['dw2'] = df['dw3'] = False
    #
    # df['KR'],df['DR'] = talib.STOCHF(df['high'].values, df['low'].values, df['close'].values, fastk_period=5, fastd_period=3, fastd_matype=0)
    # df.loc[(df['J'] < 20) & (df['pctR10MA5'] > df['pctR10MA5'].shift(1)) & (df['pctR10'] > df['pctR10MA5']), 'up1'] = True
    # df.loc[(df['J'] > 80) & (df['pctR10MA5'] < df['pctR10MA5'].shift(1)) & (df['pctR10'] < df['pctR10MA5']), 'dw1'] = True
    # df.loc[(df['KR'].shift(1) > df['DR'].shift(1)) & (df['KR'].shift(1) < 20), 'up2'] = True
    #
    # df.loc[(df['CP20'] < 0.1) & (df['CP20'] > 0) & (df['pctR10'] > df['pctR10MA5']), 'up3'] = True
    #
    # df['B'] = df['S'] = np.nan
    # df.loc[df['up1'] | df['up2'] | df['up3'] | df['1buy'], 'B'] = df['close']
    # df.loc[df['dw1'] | df['1sell'], 'S'] = df['close']

    # df['ma5'] = df['close'].rolling(5).mean()
    # df['ma10'] = df['close'].rolling(10).mean()
    # df['ma20'] = df['close'].rolling(20).mean()
    # df['ma30'] = df['close'].rolling(30).mean()
    # df['ma60'] = df['close'].rolling(60).mean()
    # df['m5m10'] = df['ma5'] >= df['ma10']
    # df['m10m20'] = df['ma10'] >= df['ma20']
    # df['m20m30'] = df['ma20'] >= df['ma30']
    # df['m30m60'] = df['ma30'] >= df['ma60']
    # for i in ['m5m10', 'm10m20', 'm20m30', 'm30m60']:
    #     df[i] = df[i].apply(lambda x: 25 if x == True else 0)
    # df1 = df[['m5m10', 'm10m20', 'm20m30', 'm30m60']]
    # df['score'] = df1.sum(axis=1).tolist()

    # return df[['datetime','B','S']]

def calcScore(data):
    pass

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
    # colors = ['green' if open_price[i] <= close_price[i] else 'red' for i in range(len(open_price))]
    # facecolors = ['green' if open_price[i] <= close_price[i] else 'white' for i in range(len(open_price))]

    line_segments1 = LineCollection(segments1, linewidths=linewidths,colors=colors)
    line_segments2 = LineCollection(segments2, linewidths=linewidths,colors=colors)
    # edgecolors = ['green' if open_price[i] <= close_price[i] else 'red' for i in range(len(open_price))]
    # bar_width = 0.25
    bar_centers = [i for i in range(len(open_price))]
    bar_bottoms = [open_price[i] if open_price[i] <= close_price[i] else close_price[i] for i in range(len(open_price))]
    # bar_bottoms = open_price if open_price <= close_price else close_price
    bar_heights = [abs(open_price[i] - close_price[i]) for i in range(len(open_price))]
    bar_segments = PolyCollection([[(x - bar_width * 1, bottom), (x - bar_width * 1, bottom + height),
                                    (x + bar_width * 1, bottom + height), (x + bar_width * 1, bottom)]
                                   for x, bottom, height in zip(bar_centers, bar_bottoms, bar_heights)],
                                  facecolors=facecolors, edgecolors=colors)
    bar_segments.set_linewidth(bar_linewidth)
                                  # facecolors=facecolors, edgecolors=['green' if open_price[i] <= close_price[i]
                                  #                                    else 'red' for i in range(len(open_price))])
    return line_segments1,line_segments2,bar_segments

class etfData(object):

    def __init__(self, api, Exapi,etfcode, etfname, bkcode, ccbcode,backset, klines, period):

        self.etfcode = etfcode  # etf代码
        self.etfname = etfname  # etf名称
        self.bkcode = bkcode    # EM板块代码
        self.ccbcode = ccbcode  # 持仓比代码
        self.backset = backset  # 起始日期
        self.klines = klines  # 结束日期
        self.period = period  #
        self.EMperiod = 1  # 为指标计算周期
        self.api = api
        self.Exapi = Exapi

    def getFullData(self):
        # ffff = self.getETFindexEM()
        df_price = self.getETFindexData()
        datestr = df_price['datetime'].values[-1][:-5]
        df_price['datetime'] = df_price['datetime'].replace(datestr+'13:00', datestr+'11:30')
        df_zjlx = self.getZJLXdata()
        df_ccb = self.getCCBdata()
        df_temp = pd.merge(df_price,df_zjlx, on=['datetime'], how='left')
        df_temp = pd.merge(df_temp,df_ccb, on=['datetime'], how='left')

        return df_temp

    def getZJLXdata(self):
        url = 'https://push2.eastmoney.com/api/qt/stock/fflow/kline/get'
        params = {'cb':'jQuery112300768268570149715_1709708951856', 'lmt':0, 'klt':1,
                'fields1':'f1,f2,f3,f7', 'fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65',
                  'ut':'b2884a393a59ad64002292a3e90d46a5','secid':'90.'+self.bkcode,'_':'1709708951857'}
        headers = {'Host': 'push2.eastmoney.com','User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0'}
        res = requests.get(url, params=params, headers=headers)

        try:
            data1 = json.loads(res.text[42:-2])['data']['klines']
        except:
            return pd.DataFrame()
        min = pd.DataFrame([i.split(',') for i in data1], columns=['datetime', 'zjlx', 'small', 'med', 'big', 'huge'])
        min.drop(labels=['small', 'med', 'big', 'huge'], axis=1, inplace=True)
        # min['time'] = min['time'].astype('datetime64[ns]')
        min['zjlx'] = min['zjlx'].astype('float') / 100000000
        min.reset_index(drop=True, inplace=True)

        return min

    def rename_stock_type_1(self,stock='600031'):
        '''
        将股票类型格式化
        stock证券代码
        1上海
        0深圳
        '''
        if stock[:3] in ['600','601','603','688','510','511',
                            '512','513','515','113','110','118','501'] or stock[:2] in ['11']:
            marker=1
        else:
            marker=0
        return marker,stock

    def getETFindexData(self):
        data = self.getETFindexTdx()
        if len(data)==0:
            data = self.getETFindexEM()
            if len(data)==0:
                print('Tdx and EM failed for ',self.etfcode)
                data = pd.DataFrame()
            return data
        else:
            return data

    def getETFindexEM(self, data_type = '1', fqt='1', limit='500', end='20500101'):
        # , stock='159805', end='20500101', limit='1000000',  data_type='D', fqt='1', count=8000):
        '''
        获取ETF基金历史数据
        stock 证券代码
        end结束时间
        limit数据长度
        data_type数据类型：
        1 1分钟
        5 5分钟
        15 15分钟
        30 30分钟
        60 60分钟
        D 日线数据
        W 周线数据
        M 月线数据
        fqt 复权
        fq=0股票除权
        fq=1前复权
        fq=2后复权
        '''

        marker, stock = self.rename_stock_type_1(self.etfcode)
        secid = '{}.{}'.format(marker, stock)
        data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
        klt = data_dict[data_type]
        params = {
            'secid': secid,
            'klt': klt,
            'fqt': fqt,
            'lmt': limit,
            'end': end,
            'iscca': '1',
            'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64',
            'ut': 'f057cbcbce2a86e2866ab8877db1d059',
            'forcect': '1',
        }
        try:
            url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?'
            res = requests.get(url=url, params=params)
            text = res.text
            json_text = json.loads(text)
            df = pd.DataFrame(json_text['data']['klines'])
            df.columns = ['数据']
            data_list = []
            for i in df['数据']:
                data_list.append(i.split(','))
            data = pd.DataFrame(data_list)
            columns = ['datetime', 'open', 'close', 'high', 'low', 'vol',
                       'amount', '_', '_', '_', '_', '_', '_', '_']
                       # 'amount', '振幅', '涨跌幅', '涨跌额', '换手率', '_', '_', '_']
            data.columns = columns
            del data['_']
            for m in columns[1:-7]:
                data[m] = pd.to_numeric(data[m])
            data1 = data.sort_index(ascending=True, ignore_index=True)
            return data1
        except:
            print('EM failed for ',self.etfcode)
            return pd.DataFrame()

    def getETFindexTdx(self, data_type='1', fqt='1', limit='1000000', end='20500101'):
        try:
            df_temp = tdxData(self.api, self.Exapi, self.etfcode, self.backset, self.klines, self.period).get_data
            # firstbar = datetime.datetime.now().strftime('%Y-%m-%d')+' 09:31'
            # df_temp = df_temp[df_temp['datetime']>=firstbar]
            columns=['datetime','open','high','low','close','vol','amount']
            df_temp.reset_index(drop=True,inplace=True)
            return df_temp[columns]
        except:
            print('tdx failed for ',self.etfcode)
            return pd.DataFrame()

    def getCCBdata(self):
        df_temp = tdxData(self.api, self.Exapi,self.ccbcode,self.backset,self.klines,self.period).get_data
        # firstbar = datetime.datetime.now().strftime('%Y-%m-%d') + ' 09:31'
        # df_temp = df_temp[df_temp['datetime'] >= firstbar]
        df_temp.rename(columns={'close':'ccb'},inplace=True)
        columns = ['datetime', 'ccb']
        df_temp.reset_index(drop=True, inplace=True)
        return df_temp[columns]

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

    def fuquan20231123(self):

        code = self.etfcode
        backset = self.backset
        qty = self.klines
        period = self.period

        fuquan = True
        zhishu = False
        if period!=9:           # 如果不是日线级别，跳过复权直接返回。
            fuquan = False
        if code[:2] in ['88','11','12','39','99','zz','zs']:  # 指数不复权
            fuquan = False

        if '#' in code:
            mkt = int(code.split('#')[0])
            code = code.split('#')[1]
            if mkt in [0,1,2] and len(code)!=6: # 深交所0 上交所1 北交所2
                print(code, 'unknown code')
                return pd.DataFrame()

        elif len(code)==6 and code[0] in '0123456789':  # A股
            if code[:2] in ['15','00','30','16','12','39','18']: # 深市
                mkt = 0 # 深交所
            elif code[:2] in ['51','58','56','60','68','50','88','11','99']:
                mkt = 1 # 上交所
            elif code[:2] in ['43','83','87']:
                mkt = 2 # 北交所
            else:
                print(code, 'unknown code')
                return pd.DataFrame()

        elif code[:2].lower()=='zz':    # 中证指数
            code = code[-6:]
            mkt=62
            fuquan = False

        elif code[:2].lower()=='zs':
            fuquan = False
            zhishu = True
            code = code[-6:]
            if code[:2] in ['39']: # 深市
                mkt = 0 # 深交所
            elif code[:2] in ['00']:
                mkt = 1 # 上交所
            elif code[:2] in ['43','83','87']:
                mkt = 2 # 北交所
            else:
                print(code, 'unknown code')
                return pd.DataFrame()

        elif len(code) == 5 and code[0]=='U':    # 期权指标
            mkt = 68

        elif len(code) == 5 and code[0]=='0':    # 港股通
            mkt = 71

        else:
            print(code, 'unknown code')
            return pd.DataFrame()

        if mkt not in [0,1,2]:

            if qty>600:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp,df_k ])
                temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
            else:
                df_k = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 0+backset, qty))

            if code[0]=='U':
                return df_k # 扩展接口不复权 直接返回数据
            else:
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

        else:

            if qty>600:
                df_k = pd.DataFrame()
                if code[:2] in ['88','99','39'] or zhishu==True:
                    for i in range(qty//600):
                        temp = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 600*i+backset, 600))
                        df_k = pd.concat([temp,df_k ])
                    temp = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                    df_k = pd.concat([temp, df_k])
                    return df_k # 指数不复权 直接返回数据
                else:
                    for i in range(qty//600):
                        temp = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 600*i+backset, 600))
                        df_k = pd.concat([temp, df_k])
                    temp = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                    df_k = pd.concat([temp, df_k])
            else:
                if code[:2] in ['88','99','39'] or zhishu==True:
                    df_k = pd.DataFrame(self.api.get_index_bars(period, mkt, code, 0+backset, qty))
                    return df_k # 指数不复权 直接返回数据
                else:
                    df_k = pd.DataFrame(self.api.get_security_bars(period, mkt, code, 0+backset, qty))

        if fuquan==False:
            return df_k

        # A股复权
        df_fuquan = self.api.get_xdxr_info(mkt, code)
        if df_fuquan is None:
            print('fuquan code no data', code)
            return df_k
        elif len(df_fuquan) == 0:
            return df_k
        else:
            df_fuquan = pd.DataFrame(df_fuquan, index=[i for i in range(len(df_fuquan))])
            df_fuquan['date'] = df_fuquan.apply(lambda x: str(x.year)+'-'+str(x.month).zfill(2)+'-'+str(x.day).zfill(2), axis=1)

            for i,row in df_fuquan.iterrows():
                if row['date'] not in list(df_k['date']):
                    continue
                elif row['name'] == '除权除息':
                    preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
                    thisclose = df_k.loc[df_k['date']==row['date'], 'close']
                    if row['fenhong']>0 and row['songzhuangu']>0:
                        change_new = (thisclose*(row['songzhuangu']/10+1)+row['fenhong']/10*0.95)/preclose-1
                        df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                    elif row['fenhong']>0:
                        change_new = (thisclose+row['fenhong']/10*0.95)/preclose-1
                        df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                    elif row['songzhuangu']>0:
                        change_new = (thisclose*(row['songzhuangu']/10+1))/preclose-1
                        df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                elif row['name'] == '扩缩股':
                    preclose = df_k.loc[df_k['date']==row['date'], 'preclose']
                    thisclose = df_k.loc[df_k['date']==row['date'], 'close']
                    change_new = (thisclose * row['suogu'])/preclose-1
                    df_k.loc[df_k['date']==row['date'], 'change'] = change_new
                elif row['name'] in ['股本变化','非流通股上市','转配股上市','送配股上市']:
                    continue
                else:
                    print(code, 'unknown name:', row['name'])
            df_k[['open', 'close', 'high', 'low']] = self.cal_right_price(df_k, type='前复权')
            return df_k

        #
        #
        # # 计算牛熊指标
        # def _Calc_func(self, x_df):
        #     '''
        #     输入为df index-date column-turnovet_rate_f|close|pct_chg
        #     ============
        #     return series/pd index-date valus-牛熊指标 kernel_n
        #     '''
        #     df = x_df.copy()
        #     periods = self.periods  # 获取计算周期
        #     if not isinstance(periods, list):
        #         periods = [periods]
        #
        #     # 计算牛熊指标
        #     series_temp = []
        #     for n in periods:
        #         turnover_ma = df['turnover_rate_f'].rolling(n).mean()
        #         std = df['pct_chg'].rolling(n).std(ddof=0)
        #         kernel_factor = std/turnover_ma
        #         kernel_factor.name = 'kernel_'+str(n)
        #         series_temp.append(kernel_factor)
        #
        #     if len(series_temp) == 1:
        #         return kernel_factor
        #
        #     else:
        #         return pd.concat(series_temp, axis=1)
        #
        # # 获取信号
        # @property
        # def get_singal(self):
        #     '''  return df index-date columns-close|pct_chg|kernel_n|turnover_rate_f    '''
        #     data = self.get_data  # 获取数据
        #     singal_df = self._Calc_func(data)  # 计算数据
        #     data = data.join(singal_df)
        #
        #     return data
        #
        # # ---------------------------------------------------------------------
        # #                   信号指标未来收益相关分析函数
        # # ---------------------------------------------------------------------
        #
        # # 计算未来收益
        # def _Cal_forward_ret(self, x_df, singal_periods):
        #     '''
        #     df为信号数据 index-date columns-close|pct_chg|kernel_n
        #     singal_periods int
        #     forwart_n 需要获取未来n日收益的list
        #     ===========
        #     return df
        #
        #     index-date colums kernel_n|未来x日收益|pct_chg
        #     '''
        #
        #     forward_n = self.forward_n  # 获取N日未来收益
        #
        #     df = x_df[['close', 'pct_chg', 'kernel_'+str(singal_periods)]].copy()
        #
        #     # 获取指数数据
        #     df['未来1日收益率'] = df['pct_chg'].shift(-1)  # next_ret
        #
        #     for n in forward_n:
        #         col = '未来{}日收益率'.format(n)
        #         df[col] = df['close'].pct_change(n).shift(-n)  # next_ret_n
        #
        #     df = df.drop('close', axis=1)  # 丢弃不需要的列
        #
        #     return df
        #
        # # 信号分段对应的收益率
        #
        # def forward_distribution_plot(self, singal_periods, bins=50):
        #     '''
        #     bins 分组
        #     =============
        #     return 图表
        #     '''
        #
        #     data = self.get_singal  # 获取信号数据
        #     forward_df = self._Cal_forward_ret(data, singal_periods)  # 获取未来收益
        #     slice_arr = pd.cut(
        #         forward_df['kernel_'+str(singal_periods)], bins=bins)  # 信号分为组
        #
        #     forward_df['barrel'] = slice_arr
        #
        #     group_ret = [x for x in forward_df.columns if x.find(
        #         '日收益率') != -1]  # 获取未来收益列
        #
        #     list_size = len(group_ret)  # 子图个数
        #
        #     # 画图
        #     f, axarr = plt.subplots(list_size, figsize=(18, 15))  # sharex=True 画子图
        #     f.suptitle(' 信号对应收益分布')  # 设置总标题
        #
        #     # 设置子图间隔等
        #     f.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.65,
        #                       wspace=0.35)
        #
        #     for i in range(list_size):
        #
        #         ret_series = forward_df.groupby('barrel')[group_ret[i]].mean()
        #         x = ret_series.index.astype('str').tolist()
        #
        #         axarr[i].bar(x, ret_series.values)  # 画子图
        #         axarr[i].set_title(group_ret[i])  # 设置子图名称
        #         axarr[i].set_xticklabels(x, rotation=90)  # 设置子图刻度标签及标签旋转
        #
        #     plt.show()
        #
        # # 画分信号位数与收益的相关系数
        # def QuantReg_plot(self, forward_ret_name, singal_periods):
        #     '''
        #     forward_ret_name 需要比较的未来收益的名称
        #     '''
        #     data = self.get_singal  # 获取信号数据
        #
        #     forward_df = self._Cal_forward_ret(data, singal_periods)  # 获取未来收益
        #
        #     x = forward_ret_name  # 设置未来收益名称
        #     traget = 'kernel_'+str(singal_periods)+'~'+x
        #
        #     mod = smf.quantreg(traget, forward_df)
        #     res = mod.fit(q=.5)
        #     print(res.summary())
        #
        #     quantiles = np.arange(.05, .96, .1)
        #
        #     def fit_model(q):
        #         res = mod.fit(q=q)
        #         # conf_int为置信区间
        #         return [q, res.params['Intercept'], res.params[x]] + \
        #                res.conf_int().loc[x].tolist()
        #
        #     models = [fit_model(q_i) for q_i in quantiles]
        #     models_df = pd.DataFrame(
        #         models, columns=['quantiles', 'Intercept', x, 'lb', 'ub'])
        #
        #     n = models_df.shape[0]
        #
        #     plt.figure(figsize=(18, 8))
        #     p1 = plt.plot(models_df['quantiles'], models_df[x],
        #                   color='red', label='Quantile Reg.')
        #     p2 = plt.plot(models_df['quantiles'], models_df['ub'],
        #                   linestyle='dotted', color='black')
        #     p3 = plt.plot(models_df['quantiles'], models_df['lb'],
        #                   linestyle='dotted', color='black')
        #
        #     plt.ylabel(r'$\beta_{RETURNS}$')
        #     plt.xlabel('信号分位数')
        #     plt.legend()
        #     plt.show()

    @property
    def get_data(self):
        # df=self.fuquan20231123(self.etfcode, self.backset, self.klines, self.period)
        df=self.fuquan20231123()
        # df['trade_date']=df['date']
        # df['pctChg']=df['close']/df['close'].shift(1)-1
        return df

def drawAllCCBmin1A():
    global backset, threshold_pct,bins,trade_rate,trendKline, cutloss, cutprofit, png_dict

    periodkey = '1分钟K线'
    period = kline_dict[periodkey]
    klines=kline_qty[periodkey]
    ma_para=20
    etf_threshold = {'上证50': 0.1582, '沪深300': 0.1705, '中证500': 0.2039, '创业板50': 0.2649}

    df_all = pd.DataFrame()

    for k,v in etf_dict.items():
        df_single = getSingleCCBData(k,period,backset, klines,ma_para)
        df_single.sort_values(by=['datetime'],ascending=True, inplace=True)
        df_single.reset_index(drop=True, inplace=True)
        # df_single['time'] = df_single['datetime'].apply(lambda x: x.split(' ')[1])
        df_single['pctChg'] = df_single['close']/df_single['close'].shift(1)-1

        df_single['close'] = df_single['close'].ffill()
        df_single['cm5'] = df_single['close'].rolling(5).mean()
        df_single['cm20'] = df_single['close'].rolling(20).mean()
        df_single['cmgap'] = (df_single['cm5'] - df_single['cm20'])/df_single['cm5']

        df_single['gap'] = (df_single['close'] - df_single['cm20'])/df_single['close']*100
        df_single['gapabs'] = df_single['gap'].apply(lambda x: abs(x))
        gap_threshold = etf_threshold[k]
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
        # df_single['ccbgap'] = df_single['cmgap']-df_single['ccbmgap']
        df_single['ccbgapm20'] = df_single['ccbgap'].rolling(20).mean()
        df_single.loc[(df_single['ccbgap']>df_single['ccbgapm20']) & (df_single['mark']>=0),'up2'] = 0  # (df_single['mark']>=0) &
        df_single.loc[(df_single['ccbgap']<df_single['ccbgapm20']) & (df_single['mark']<=0),'dw2'] = 0

        df_single.loc[(df_single['ccbgap']>df_single['ccbgapm20']) & (df_single['ccbgap'].shift(1)<df_single['ccbgapm20'].shift(1)),'up'] = -1.5  # (df_single['mark']>=0) &
        df_single.loc[(df_single['ccbgap']<df_single['ccbgapm20']) & (df_single['ccbgap'].shift(1)>df_single['ccbgapm20'].shift(1)),'dw'] = -1.5

        df_single['etf'] = k
        df_all = pd.concat([df_all, df_single])

    df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'ccbh', 'ccbl', 'close', 'high', 'low','open',
               'gap','gapSig','cm20','cmgap','ccp60','ccbcp60','ccbgap','ccbgapm20','ccbmgap','up','dw','up2','dw2'], dropna=False)
    df_pivot.reset_index(drop=False,inplace=True)

    df_pivot['time'] = df_pivot[('datetime','')].apply(lambda x: x.split(' ')[1])
    firstbar = df_pivot[('time','')].tolist().index('15:00')
    df_pivot = df_pivot[firstbar:]

    # df_whole = df_pivot.copy()
    df_pivot = df_pivot[-241:]

    df_pivot.reset_index(drop=True,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)
    openbar = [id for id,v in enumerate(df_pivot[('datetime')].tolist()) if '09:31' in v][0]-1

    df_pivot['zero'] = 0
    lastBar = df_pivot[('datetime','')].values[-1].replace('-','').replace(':','').replace(' ','_')

    fig, ax = plt.subplots(4, 1, figsize=(18,9), sharex=True)

    # else:
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

        # x.plot(df_pivot.index, df_pivot[('close',k)], label=k, linewidth=1, color='red',marker='.', markersize=4,alpha=0.7)
        x.plot(df_pivot.index, df_pivot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='-.', color='red', alpha=1.)
        x.vlines(openbar, ymin=df_pivot[('close', k)].min(), ymax=df_pivot[('close', k)].max(), color='blue', linestyles='--',alpha=1)

        x3 = x.twinx()
        # x3.bar(df_pivot.index, df_pivot[('ccbmgap',k)], width=1.0, color='green',alpha=0.15,zorder=-20)
        # x3.bar(df_pivot.index, df_pivot[('cmgap',k)], width=1.0, color='red',alpha=0.15,zorder=-20)
        # x3.bar(df_pivot.index,df_pivot[('ccbmgap',k)],label='ccbgap', linewidth=0.7, color='green',alpha=1,zorder=-10)
        x3.set_yticks([])

        x4 = x.twinx()
        x4.plot(df_pivot.index,df_pivot[('ccb',k)],label='ccb', linewidth=0.7, color='green',alpha=1,zorder=-10)
        x4.set_yticks([])
        #
        x5 = x.twinx()
        # x5.plot(df_pivot.index,df_pivot[('ccp60',k)],  color='red',linewidth=0.7,linestyle='--')
        # x5.plot(df_pivot.index,df_pivot[('ccbcp60',k)], color='green',linewidth=0.7,linestyle='--')
        x5.plot(df_pivot.index,df_pivot[('ccbgap',k)],  color='blue',linewidth=1.0,linestyle='-')  #marker='.',
        x5.plot(df_pivot.index,df_pivot[('ccbgapm20',k)], color='blue',linestyle='-', linewidth=0.6)
        x5.scatter(df_pivot.index, df_pivot[('up', k)], marker='^', s=36, c='red',alpha=0.7,zorder=-30)
        x5.scatter(df_pivot.index, df_pivot[('dw', k)], marker='v',s=36, c='green',alpha=0.7,zorder=-30)
        x5.scatter(df_pivot.index, df_pivot[('up2', k)], s=25, c='r', marker='s', alpha=0.3,zorder=-20)
        x5.scatter(df_pivot.index, df_pivot[('dw2', k)], s=25, c='g', marker='s', alpha=0.3,zorder=-20)
        x5.hlines(0, xmin=df_pivot.index.min(), xmax=df_pivot.index.max(), color='k',linewidth=0.5,alpha=0.5,zorder=-25)
        x5.set_ylim(-2,2)
        # x5.set_yticks([])

        x6 = x.twinx()
        x6.plot(df_pivot.index,df_pivot[('gap',k)], linewidth=0.5, color='violet',linestyle='dotted',zorder=-28,alpha=1)
        x6.plot(df_pivot.index,df_pivot[('gapSig',k)], linewidth=2, color='violet', zorder=-30,alpha=0.6)
        x6.set_yticks([])

        x.legend(loc='upper left')
        x4.legend(loc='center left')
        # x5.legend(loc='lower left')

        x.minorticks_on()
        x.grid(which='major', axis="x", color='k', linestyle='-', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='--', linewidth=0.15)

        if k in png_dict.keys():
            x.text(0.25,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        x.text(0.1, 1., k, horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.suptitle(lastBar,x=0.6, y=0.98)
    plt.savefig('output\\持续监控ETF1分钟2.0a_' + datetime.datetime.now().strftime('%Y%m%d') + '.png')

    fig.clf()
    plt.close(fig)

def drawAllCCBmin1B():
    global backset, threshold_pct,bins,trade_rate,trendKline, cutloss, cutprofit, png_dict

    periodkey = '1分钟K线'
    period = kline_dict[periodkey]
    klines=kline_qty[periodkey]
    ma_para=kline_mapara[periodkey]
    etf_threshold = {'上证50': 0.1582, '沪深300': 0.1705, '中证500': 0.2039, '创业板50': 0.2649}

    df_all = pd.DataFrame()

    for k,v in etf_dict.items():
        df_single = getSingleCCBData(k,period,backset, klines,ma_para)
        df_single.sort_values(by=['datetime'],ascending=True, inplace=True)
        df_single.reset_index(drop=True, inplace=True)
        df_single['time'] = df_single['datetime'].apply(lambda x: x.split(' ')[1])
        df_single['pctChg'] = df_single['close']/df_single['close'].shift(1)-1

        # df_single['aft01d'] = df_single['close'].shift(-1)/df_single['close']*10000-10000
        # df_single['aft03d'] = df_single['close'].shift(-3)/df_single['close']*10000-10000
        # df_single['aft05d'] = df_single['close'].shift(-5)/df_single['close']*10000-10000
        # df_single['aft10d'] = df_single['close'].shift(-10)/df_single['close']*10000-10000
        # df_single['aft20d'] = df_single['close'].shift(-20)/df_single['close']*10000-10000
        # df_single['aft30d'] = df_single['close'].shift(-30)/df_single['close']*10000-10000


        #####  评分系统 - CCB
        df_single['ccbma5'] = df_single['ccb'].rolling(5).mean()
        df_single['ccbma20'] = df_single['ccb'].rolling(20).mean()
        df_single['ccbma100'] = df_single['ccb'].rolling(100).mean()
        df_single['ccbma100'] = df_single['ccb'].rolling(100).mean()
        df_single['ccbabv100'] = (df_single['ccb']>df_single['ccbma100']).map({True:1,False:-1})
        df_single['ccbabv100accum'] = barsAccum(df_single['ccbabv100'].tolist())
        df_single['ccbabv100accumabs'] = df_single['ccbabv100accum'].apply(lambda x: abs(x))
        ccbabv100accumThreshold = get_threshold(df_single['ccbabv100accumabs'].dropna(), bins, threshold_pct)
        # ccbabv100accumThreshold2= get_my_threshold(df_single['ccbabv100'].dropna().tolist(),bins, threshold_pct)
        df_single['ccbturn'] = df_single['ccbmark'] = 0
        df_single.loc[(df_single['ccbabv100accum']==1) & (df_single['ccbabv100accum'].shift(1)<-ccbabv100accumThreshold),
            'ccbturn'] = -1
        df_single.loc[(df_single['ccbabv100accum']==-1) & (df_single['ccbabv100accum'].shift(1)>ccbabv100accumThreshold),
            'ccbturn'] = 1
        df_single.loc[(df_single['ccbturn'].rolling(3).sum()!=0), 'ccbmark'] = df_single['ccbturn'].rolling(3).sum()


        #####  评分系统 - 价格偏离度
        df_single['cm5'] = df_single['close'].rolling(5).mean()
        df_single['cm20'] = df_single['close'].rolling(20).mean()
        df_single['cmax120'] = df_single['close'].rolling(120).max()
        df_single['cmax240'] = df_single['close'].rolling(240).max()
        df_single['cmin120'] = df_single['close'].rolling(120).min()
        df_single['cmin240'] = df_single['close'].rolling(240).min()

        df_single['gap'] = (df_single['close'] - df_single['cm20'])/df_single['close']*100
        df_single['gapabs'] = df_single['gap'].apply(lambda x: abs(x))
        # gap_threshold = plot_hist(df_single["gapabs"], bins, threshold_pct)
        # gap_threshold = get_threshold(df_single['gapabs'].dropna(), bins, threshold_pct)
        gap_threshold = etf_threshold[k]
        df_single['cmark'] = 0
        df_single.loc[(df_single['gap']>gap_threshold),'gapSig'] = df_single['gap']
        df_single.loc[(df_single['gap']<-1*gap_threshold),'gapSig'] = df_single['gap']
        df_single['cturn'] = df_single['cmark'] = 0
        df_single.loc[(df_single['close']>df_single['close'].shift(1)) & (df_single['gapSig']<0), 'cturn'] = 1
        df_single.loc[(df_single['close']<df_single['close'].shift(1)) & (df_single['gapSig']>0), 'cturn'] = -1
        df_single.loc[(df_single['cturn'].rolling(1).sum()!=0), 'cmark'] = df_single['cturn'].rolling(1).sum()

        ######  评分系统 - 信号
        # 期权趋势交易机会
        # 1a 大趋势下跌，小幅反弹后再次跌破ma5, 止损位 反弹的高点
        # 1b 大趋势上涨，小幅下跌后再次上穿ma5，止损位 下跌的地点
        df_single.loc[df_single['close'] > df_single['close'].shift(trendKline), 'trend'] = 1
        df_single.loc[df_single['close'] < df_single['close'].shift(trendKline), 'trend'] = -1
        df_single.loc[(df_single['close'] > df_single['cm5']) & (df_single['close'].shift(1) < df_single['cm5'].shift(1)), 'ccrossm5'] = 1
        df_single.loc[(df_single['close'] < df_single['cm5']) & (df_single['close'].shift(1) > df_single['cm5'].shift(1)), 'ccrossm5'] = -1
        df_single.loc[(df_single['trend'] == 1) & (df_single['ccrossm5']==1) & (df_single['close']<df_single['cm20']), 'trendbuy'] = df_single['close']    #
        df_single.loc[(df_single['trend'] == -1) & (df_single['ccrossm5']==-1)  & (df_single['close']>df_single['cm20']), 'trendsell'] = df_single['close']   # (df_single['close'] > df_single['cm20'])
        # 期权回归交易机会
        # 1a 底部反弹 gap<-阈值，c>ma5，止损位 最近的最低位
        # 1b 顶部下坠:  gap>阈值，c<ma5，止损位 最近的最高位
        df_single['gapmark'] = 0
        df_single.loc[df_single['gap'] > gap_threshold, 'gapmark'] = -1
        df_single.loc[df_single['gap'] < -1*gap_threshold, 'gapmark'] = 1
        df_single.loc[(df_single['ccrossm5'] == 1) & (df_single['gapmark'].rolling(5).sum()>0) & (df_single['close']<df_single['cm20']), 'reversebuy'] = df_single['close']
        df_single.loc[(df_single['ccrossm5'] == -1) & (df_single['gapmark'].rolling(5).sum()<0) & (df_single['close']>df_single['cm20']), 'reversesell'] = df_single['close']
        # df_single.reset_index(drop=True,inplace=True)
        df_single['flag'] = df_single['flagmark'] = 0
        df_single['reason'] = ''
        df_single.loc[(df_single['time'] > '09:29') & (df_single['time'] <'14:56'), 'timeopentrade'] = 1
        # df_single.loc[(df_single['reversebuy']>0) & (df_single['timeopentrade']==1), 'flag'] = 1
        # df_single.loc[(df_single['reversesell']>0) & (df_single['timeopentrade']==1), 'flag'] = -1
        df_single.loc[((df_single['reversebuy']>0) | (df_single['trendbuy']>0)) & (df_single['timeopentrade']==1), 'flag'] = 1
        df_single.loc[((df_single['reversesell']>0) | (df_single['trendsell']>0)) & (df_single['timeopentrade']==1), 'flag'] = -1
        df_single.loc[(df_single['reversebuy']>0) & (df_single['timeopentrade']==1) , 'reason'] = '底部买'
        df_single.loc[(df_single['reversesell']>0) & (df_single['timeopentrade']==1), 'reason'] = '顶部卖'
        df_single.loc[(df_single['trendbuy']>0) & (df_single['timeopentrade']==1), 'reason'] = '顺势买'
        df_single.loc[(df_single['trendsell']>0) & (df_single['timeopentrade']==1), 'reason'] = '顺势卖'
        df_single.loc[(df_single['flag'].rolling(1).sum()!=0), 'flagmark'] = df_single['flag'].rolling(1).sum()

        #####  当前K线总体评分 = ccbmark + cmark + flagmark
        df_single['allmarks'] = df_single['ccbmark'] #+ df_single['cmark'] + df_single['flagmark']

        df_single['etf'] = k

        df_all = pd.concat([df_all, df_single])


    # df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'ccbh', 'ccbl', 'close', 'high', 'low','open',
    #            'cm5','cm20','ccbma5','ccbma20','gap','gapSig','trendbuy','trendsell','reversebuy','reversesell',
    #            'trxOpen', 'trxClose', 'action','flag','curve'], dropna=False)

    df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'ccbh', 'ccbl', 'close', 'high', 'low','open',
               'cm5','cm20','ccbma5','ccbma20','ccbma100','gap','gapSig','trendbuy','trendsell','reversebuy','reversesell',
               'allmarks','ccbmark','cmark','flagmark','cmax120','cmax240','cmin120','cmin240'], dropna=False)
    # df_labels = df_all.pivot_table(index='datetime',columns='etf',values=['reason'], dropna=False,aggfunc=original_strings, fill_value='')
    # df_pivot = pd.merge(df_pivot, df_labels, left_index=True, right_index=True)

    df_pivot.reset_index(drop=False,inplace=True)
    # df_pivot.reset_index(drop=False,inplace=True)

    df_pivot['time'] = df_pivot[('datetime','')].apply(lambda x: x.split(' ')[1])
    firstbar = df_pivot[('time','')].tolist().index('15:00')
    df_pivot = df_pivot[firstbar:]

    df_whole = df_pivot.copy()

    # for kbars in range(240*1, len(df_pivot), 240):
    df_pivot = df_whole[-241:]
    # if len(df_pivot)<48:
    #     continue

    df_pivot.reset_index(drop=True,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)
    openbar = [id for id,v in enumerate(df_pivot[('datetime')].tolist()) if '09:31' in v][0]-1

    df_pivot['zero'] = 0
    lastBar = df_pivot[('datetime','')].values[-1].replace('-','').replace(':','').replace(' ','_')

    fig, ax = plt.subplots(4, 1, figsize=(19,9), sharex=True)
    # plt.suptitle('ETF期权持仓比  时间戳:' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))

    # else:
    tickGap = 30
    xlables = [i[5:].replace('-','').replace(':','').replace(' ','_') for i in df_pivot[( 'datetime','')][::tickGap]]
    plt.xticks(range(len(df_pivot))[::tickGap],xlables,rotation=90)

    for x,k in zip(ax,  etf_dict.keys()):
    # for x,k in zip([ax],  ['上证50']):

        dftemp = df_pivot[[('index', ''), ('open', k), ('close', k), ('high', k), ('low', k)]]
        dftemp.columns = ['index', 'open', 'close', 'high', 'low']
        # line_seg1, line_seg2, bar_segments = getKlineObjects(dftemp, bar_width=0.25)
        # x.add_collection(line_seg1)
        # x.add_collection(line_seg2)
        # x.add_collection(bar_segments)
        line_seg1, line_seg2, line_seg3 = construct_ohlc_collections(dftemp, wid=0.4)
        x.add_collection(line_seg1)
        x.add_collection(line_seg2)
        x.add_collection(line_seg3)

        # x.plot(df_pivot.index, df_pivot[('close',k)], label=k, linewidth=1, color='red',marker='.', markersize=4,alpha=0.7)
        x.plot(df_pivot.index, df_pivot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='-.', color='red', alpha=1.)
        x.plot(df_pivot.index, df_pivot[('cm5', k)], label='ma5', linewidth=0.9, linestyle='dotted', color='red', alpha=1.)
        # x.plot(df_pivot.index, df_pivot[('cmax120', k)], linewidth=0.8,  color='red', alpha=1.)
        # x.plot(df_pivot.index, df_pivot[('cmax240', k)], linewidth=3,  color='green', alpha=0.5)
        # x.plot(df_pivot.index, df_pivot[('cmin120', k)], linewidth=0.8, color='red', alpha=1.)
        # x.plot(df_pivot.index, df_pivot[('cmin240', k)], linewidth=3, color='green', alpha=0.5)
        x.vlines(openbar, ymin=df_pivot[('close', k)].min(), ymax=df_pivot[('close', k)].max(), color='blue',linestyles='--', alpha=1)

        x.scatter(df_pivot.index, df_pivot[('trendbuy', k)]*0.998, s=49, c='r', marker='^', alpha=0.5)
        x.scatter(df_pivot.index, df_pivot[('trendsell', k)]*1.002, s=49, c='g', marker='v', alpha=0.5)
        x.scatter(df_pivot.index, df_pivot[('reversebuy', k)] * 0.998, s=36, c='r', marker='D', alpha=0.5)
        x.scatter(df_pivot.index, df_pivot[('reversesell', k)] * 1.002, s=36, c='g', marker='D', alpha=0.5)


        x5 = x.twinx()
        x5.plot(df_pivot.index,df_pivot[('gap',k)], linewidth=0.5, color='black',linestyle='dotted',zorder=-28,alpha=1)
        x5.plot(df_pivot.index,df_pivot[('gapSig',k)], linewidth=2, color='black', zorder=-30,alpha=0.4)
        x5.set_yticks([])

        x.legend(loc='upper left')
        # x2.legend(loc='center left')
        # x4.legend(loc='lower left')

        x.minorticks_on()
        x.grid(which='major', axis="x", color='k', linestyle='-', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='--', linewidth=0.15)

        if k in png_dict.keys():
            x.text(0.25,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        x.text(0.25, 0.9, k, horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.suptitle(lastBar,x=0.6, y=0.98)

    plt.savefig('output\\持续监控ETF1分钟2.0b_' + datetime.datetime.now().strftime('%Y%m%d') + '.png')

    fig.clf()
    plt.close(fig)

def drawAllCCBmin1C(bars=181):
    global png_dict

    df_final = pd.DataFrame()
    etfbk_dict = {'上证50':'BK0611','沪深300':'BK0500','中证500':'BK0701','创业板50':'BK0638'}
    etf_threshold = {'上证50':0.1582,'沪深300':0.1705,'中证500':0.2039,'创业板50':0.2649}

    periodkey = '1分钟K线'
    period = kline_dict[periodkey]
    klines=kline_qty[periodkey]
    trendKline = 240


    for key in etf_dict.keys():
        etfcode = etfcode_dict[key]
        etfbkcode = etfbk_dict[key]
        ccbcode = etf_ccb_dict[key]
        gap_threshold =  etf_threshold[key]

        temp = etfData(api, Exapi,etfcode, key, etfbkcode,ccbcode,backset, klines, period)
        df_single = temp.getFullData()
        # today = df_single['datetime'].values[-1]
        df_single['cm5'] = df_single['close'].rolling(5).mean()
        df_single['cm20'] = df_single['close'].rolling(20).mean()

        df_single['ccbm10'] = df_single['ccb'].rolling(20).mean()
        df_single['zjlxm10'] = df_single['zjlx'].rolling(15).mean()

        df_single['gap'] = (df_single['close'] - df_single['cm20'])/df_single['close']*100
        df_single['gapabs'] = df_single['gap'].apply(lambda x: abs(x))
        df_single.loc[(df_single['gap']>gap_threshold),'gapSig'] = df_single['gap']
        df_single.loc[(df_single['gap']<-1*gap_threshold),'gapSig'] = df_single['gap']

        # 期权趋势交易机会
        # 1a 大趋势下跌，小幅反弹后再次跌破ma5, 止损位 反弹的高点
        # 1b 大趋势上涨，小幅下跌后再次上穿ma5，止损位 下跌的地点
        df_single.loc[df_single['close'] > df_single['close'].shift(trendKline), 'trend'] = 1
        df_single.loc[df_single['close'] < df_single['close'].shift(trendKline), 'trend'] = -1
        df_single.loc[(df_single['close'] > df_single['cm5']) & (df_single['close'].shift(1) < df_single['cm5'].shift(1)), 'ccrossm5'] = 1
        df_single.loc[(df_single['close'] < df_single['cm5']) & (df_single['close'].shift(1) > df_single['cm5'].shift(1)), 'ccrossm5'] = -1
        df_single.loc[(df_single['trend'] == 1) & (df_single['ccrossm5']==1) & (df_single['close']<df_single['cm20']), 'trendbuy'] = df_single['close']    #
        df_single.loc[(df_single['trend'] == -1) & (df_single['ccrossm5']==-1)  & (df_single['close']>df_single['cm20']), 'trendsell'] = df_single['close']   # (df_single['close'] > df_single['cm20'])


        # 期权回归交易机会
        # 1a 底部反弹 gap<-阈值，c>ma5，止损位 最近的最低位
        # 1b 顶部下坠:  gap>阈值，c<ma5，止损位 最近的最高位
        df_single['gapmark'] = 0
        df_single.loc[df_single['gap'] > gap_threshold, 'gapmark'] = -1
        df_single.loc[df_single['gap'] < -1*gap_threshold, 'gapmark'] = 1
        df_single.loc[(df_single['ccrossm5'] == 1) & (df_single['gapmark'].rolling(5).sum()>0) & (df_single['close']<df_single['cm20']), 'reversebuy'] = df_single['close']
        df_single.loc[(df_single['ccrossm5'] == -1) & (df_single['gapmark'].rolling(5).sum()<0) & (df_single['close']>df_single['cm20']), 'reversesell'] = df_single['close']


        df_single.loc[(df_single['ccb']>df_single['ccbm10']) & (df_single['ccb'].shift(1)<df_single['ccbm10'].shift(1)), 'ccbsell'] = df_single['ccb']
        df_single.loc[(df_single['ccb']<df_single['ccbm10']) & (df_single['ccb'].shift(1)>df_single['ccbm10'].shift(1)), 'ccbbuy'] = df_single['ccb']

        df_single.loc[(df_single['zjlx']>df_single['zjlxm10']) & (df_single['zjlx'].shift(1)<df_single['zjlxm10'].shift(1)), 'zjlxbuy'] = df_single['zjlx']
        df_single.loc[(df_single['zjlx']<df_single['zjlxm10']) & (df_single['zjlx'].shift(1)>df_single['zjlxm10'].shift(1)), 'zjlxsell'] = df_single['zjlx']

        df_single.loc[(df_single['zjlx']>df_single['zjlxm10']) & (df_single['ccb']<df_single['ccbm10']) & \
            (df_single['ccbbuy'].notnull() | df_single['zjlxbuy'].notnull()), 'buy'] = df_single['close']
        df_single.loc[(df_single['zjlx']<df_single['zjlxm10']) & (df_single['ccb']>df_single['ccbm10']) & \
            (df_single['ccbsell'].notnull() | df_single['zjlxsell'].notnull()), 'sell'] = df_single['close']

        df_single = df_single[-bars:]

        df_single['name'] = key
        df_final = pd.concat([df_final, df_single])

    df_pivot = df_final.pivot_table(index='datetime',columns='name',values=['open', 'high', 'low', 'close', 'vol', 'amount', 'zjlx',
       'ccb', 'cm5', 'cm20', 'ccbm10', 'zjlxm10', 'ccbsell', 'ccbbuy','zjlxbuy', 'zjlxsell', 'buy', 'sell',
        'trendbuy', 'trendsell', 'reversebuy', 'reversesell','gap','gapSig'], dropna=False)#aggfunc=original_strings, fill_value='')

    df_pivot.reset_index(drop=False,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)

    df_pivot['zero'] = 0
    openbar = [id for id, v in enumerate(df_pivot[('datetime')].tolist()) if ':00' in v][0]
    lastBar = df_pivot[('datetime','')].values[-1].replace('-','').replace(':','').replace(' ','_')
    # fig, ax = plt.subplots(4, 1, figsize=(19,9), sharex=True)
    # plt.suptitle('ETF期权持仓比  时间戳:' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    tickGap = 30
    xlables = [i[5:].replace('-','').replace(':','').replace(' ','_') for i in df_pivot[( 'datetime','')][::tickGap]]

    fig, ax = plt.subplots(2, 2, figsize=(18, 9),sharex=False)

    plt.xticks(range(len(df_pivot))[::tickGap],xlables,rotation=90)
    ax[0,0].set_xticks(range(len(df_pivot))[::tickGap])
    ax[0,0].set_xticklabels(xlables)
    ax[0,1].set_xticks(range(len(df_pivot))[::tickGap])
    ax[0,1].set_xticklabels(xlables)
    ax[1,0].set_xticks(range(len(df_pivot))[::tickGap])
    ax[1,0].set_xticklabels(xlables)
    ax[1,1].set_xticks(range(len(df_pivot))[::tickGap])
    ax[1,1].set_xticklabels(xlables)

    for idx,k in enumerate(etf_dict.keys()):
        if idx==0:
            x = ax[0,0]
        elif idx==1:
            x = ax[0,1]
        elif idx==2:
            x = ax[1,0]
        elif idx==3:
            x = ax[1,1]
        else:
            print('no ax for ',idx)

        dftemp = df_pivot[[('index',''),('open',k),('close',k),('high',k),('low',k)]]
        dftemp.columns=['index','open','close','high','low']
        # line_seg1,line_seg2,bar_segments = getKlineObjects(dftemp,bar_width=0.25)
        line_seg1,line_seg2, line_seg3 = construct_ohlc_collections(dftemp,wid=0.4)
        x.add_collection(line_seg1)
        x.add_collection(line_seg2)
        x.add_collection(line_seg3)

        x.plot(df_pivot.index, df_pivot[('cm20',k)], label='ma20', linewidth=0.7, linestyle='-.', color='red', alpha=1.)
        x.vlines(openbar, ymin=df_pivot[('close', k)].min(), ymax=df_pivot[('close', k)].max(), color='blue', linestyles='--',alpha=1)

        x.scatter(df_pivot.index, df_pivot[('trendbuy', k)] * 0.998, s=49, c='r', marker='^', alpha=0.6)
        x.scatter(df_pivot.index, df_pivot[('trendsell', k)] * 1.002, s=49, c='g', marker='v', alpha=0.6)
        x.scatter(df_pivot.index, df_pivot[('reversebuy', k)] * 0.998, s=36, c='r', marker='D', alpha=0.6)
        x.scatter(df_pivot.index, df_pivot[('reversesell', k)] * 1.002, s=36, c='g', marker='D', alpha=0.6)

        x2 = x.twinx()
        x2.plot(df_pivot.index, df_pivot[('ccb',k)], label='ccb', linewidth=0.8, color='green')
        x2.plot(df_pivot.index, df_pivot[('ccbm10',k)], label='ccbm10', linewidth=0.9, linestyle='dotted',color='green')
        x2.set_yticks([])

        x3 = x.twinx()
        x3.plot(df_pivot.index, df_pivot[('zjlx',k)], label='zjlx', linewidth=0.7, color='blue')
        x3.plot(df_pivot.index, df_pivot[('zjlxm10',k)], label='zjlxm10', linewidth=0.9, linestyle='dotted', color='blue')
        x3.hlines(df_pivot[('zjlx',k)].mean(),xmin=df_pivot.index.min(), xmax=df_pivot.index.max(),color='white',zorder=-20, alpha=0.1)
        # x3.scatter(df_pivot.index,df_pivot[('gapdw',k)], s=4, c='g',marker='.',alpha=0.8)

        x4 = x.twinx()
        x4.bar(df_pivot.index, df_pivot[('amount',k)], width=1.0, color='blue',alpha=0.1,zorder=-10)
        x4.set_yticks([])

        x5 = x.twinx()
        x5.plot(df_pivot.index, df_pivot[('gap', k)], linewidth=0.5, color='black', linestyle='dotted',zorder=-28, alpha=0.8)
        x5.plot(df_pivot.index, df_pivot[('gapSig', k)], linewidth=2, color='black', zorder=-30, alpha=0.4)
        x5.set_yticks([])

        x.legend(loc='upper left')
        x2.legend(loc='center left')
        x3.legend(loc='lower left')

        x.minorticks_on()
        x.grid(which='major', axis="both", color='k', linestyle='-', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='--', linewidth=0.15)
        if k in png_dict.keys():
            # x.text(0.25,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')
            x.set_title(png_dict[k], x=0.5, y=0.98)
        else:
            x.set_title(k, x=0.5, y=0.98)

    plt.tight_layout()
    plt.suptitle(datetime.datetime.now().strftime('%Y%m%d'), x=0.5, y=0.98)
    plt.savefig('output\\持续监控ETF1分钟2.0c_'+ datetime.datetime.now().strftime('%Y%m%d')+'.png')
    fig.clf()
    plt.close(fig)

def myStrategy(data, gap_threshold, cutloss=-0.005, cutprofit=0.01):

    df = data.copy()

    if 'ccb' not in df.columns:
        print('ccb not in column')
        return pd.DataFrame()


    df.loc[0, "flag"] = 0
    df['remark'] = ''
    df['action'] = df['winpct'] = df['winpctMax'] = 0
    df['trxOpen'] = df['trxClose'] = df['trxBars'] = 0
    status = trxOpen = barOpen = winpct = winpctMax = 0
    lastTrx = lastOpen = lastGain = lastTrxIdx = 0

    for idx in range(1, len(df), 1):
        # 空仓时
        if status == 0 and df['flag'].values[idx] == 1:
            df.iloc[idx, df.columns.get_loc('remark')] = '多头开仓'
            df.iloc[idx, df.columns.get_loc('action')] = 1
            df.iloc[idx, df.columns.get_loc('RET')] = -1 * trade_rate
            status = 1
            openReason = df['reason'].values[idx]
            barOpen = idx
            trxOpen = df['close'].values[idx]
            df.iloc[idx, df.columns.get_loc('trxOpen')] = trxOpen
        elif status == 0 and df['flag'].values[idx] == -1:
            df.iloc[idx, df.columns.get_loc('remark')] = '空头开仓'
            df.iloc[idx, df.columns.get_loc('action')] = -1
            df.iloc[idx, df.columns.get_loc('RET')] = -1 * trade_rate
            status = -1
            openReason = df['reason'].values[idx]
            barOpen = idx
            trxOpen = df['close'].values[idx]
            df.iloc[idx, df.columns.get_loc('trxOpen')] = trxOpen
        elif status == 0 and df['flag'].values[idx] == 0:
            df.iloc[idx, df.columns.get_loc('remark')] = '空仓保持'
            pass
        # 持有多头仓位时
        elif status == 1:
            winpct = df['close'].values[idx] / trxOpen - 1
            winpctMax = max(winpct, winpctMax)
            longExit1 = df["gap"].values[idx] > gap_threshold and idx > barOpen + 5  # 上涨过头
            longExit2 = df["cm5"].values[idx] < df["cm5"].values[idx - 1] and \
                        df['close'].values[idx] < df['cm5'].values[idx]  # MA5下拐 belowMA5
            longExit3 = df["cm5_20"].values[idx] < df["cm5_20"].values[idx - 1] and \
                        df['cm5_20'].values[idx - 1] < df['cm5_20'].values[idx - 2]  # cm5_20 持续减小
            longExit = longExit1  # or longExit2
            if winpct < cutloss:  # 多头止损
                lastTrx = 1
                lastOpen = trxOpen
                lastGain = winpct
                lastTrxIdx = idx
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '多头止损'
                df.iloc[idx, df.columns.get_loc('reason')] = '止损'
                df.iloc[idx, df.columns.get_loc('action')] = 2
                df.iloc[idx, df.columns.get_loc('RET')] = df['pctChg'].values[idx] - trade_rate
                df.iloc[idx, df.columns.get_loc('trxClose')] = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = 0
            elif winpct > cutprofit:  # 多头止盈
                lastTrx = 1
                lastOpen = trxOpen
                lastGain = winpct
                lastTrxIdx = idx
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '多头止盈'
                df.iloc[idx, df.columns.get_loc('reason')] = '止盈'
                df.iloc[idx, df.columns.get_loc('action')] = 2
                df.iloc[idx, df.columns.get_loc('RET')] = df['pctChg'].values[idx] - trade_rate
                df.iloc[idx, df.columns.get_loc('trxClose')] = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = 0
            elif longExit:  # 上涨过头平仓
                lastTrx = 1
                lastOpen = trxOpen
                lastGain = winpct
                lastTrxIdx = idx
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '上涨过头平仓'
                df.iloc[idx, df.columns.get_loc('reason')] = '涨过头'
                df.iloc[idx, df.columns.get_loc('action')] = 2
                df.iloc[idx, df.columns.get_loc('RET')] = df['pctChg'].values[idx] - trade_rate
                df.iloc[idx, df.columns.get_loc('trxClose')] = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = 0
            elif df['flag'].values[idx] in [0, 1]:  # 多头持仓
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '多头持仓'
                df.iloc[idx, df.columns.get_loc('action')] = 1
                df.iloc[idx, df.columns.get_loc('RET')] = df['pctChg'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
            elif df['flag'].values[idx] == -1:  # 多头转空头
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '多头转空头'
                df.iloc[idx, df.columns.get_loc('reason')] = '多转空'
                df.iloc[idx, df.columns.get_loc('action')] = 0
                df.iloc[idx, df.columns.get_loc('RET')] = df['pctChg'].values[idx] - 2 * trade_rate
                trxOpen = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxOpen')] = trxOpen
                df.iloc[idx, df.columns.get_loc('trxClose')] = trxOpen
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = -1
                openReason = df['reason'].values[idx]
                barOpen = idx

        # 持有空头仓位时
        elif status == -1:
            winpct = 1 - df['close'].values[idx] / trxOpen
            winpctMax = max(winpct, winpctMax)
            shortExit1 = df["gap"].values[idx] < -gap_threshold and idx > barOpen + 5  # 下跌过头
            shortExit2 = df["cm5"].values[idx] > df["cm5"].values[idx - 1] and \
                         df['close'].values[idx] > df['cm5'].values[idx]  # MA5上拐 abvMA5
            shortExit3 = df["cm5_20"].values[idx] > df["cm5_20"].values[idx - 1] and \
                         df['cm5_20'].values[idx - 1] > df['cm5_20'].values[idx - 2]  # cm5_20 持续变大
            shortExit = shortExit1
            if winpct < cutloss:  # 空头止损
                lastTrx = -1
                lastOpen = trxOpen
                lastGain = winpct
                lastTrxIdx = idx
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '空头止损'
                df.iloc[idx, df.columns.get_loc('reason')] = '止损'
                df.iloc[idx, df.columns.get_loc('action')] = -2
                df.iloc[idx, df.columns.get_loc('RET')] = -df['pctChg'].values[idx] - trade_rate
                df.iloc[idx, df.columns.get_loc('trxClose')] = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = 0
            elif winpct > cutprofit:  # 空头止盈
                lastTrx = -1
                lastOpen = trxOpen
                lastGain = winpct
                lastTrxIdx = idx
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '空头止盈'
                df.iloc[idx, df.columns.get_loc('reason')] = '止盈'
                df.iloc[idx, df.columns.get_loc('action')] = -2
                df.iloc[idx, df.columns.get_loc('RET')] = -df['pctChg'].values[idx] - trade_rate
                df.iloc[idx, df.columns.get_loc('trxClose')] = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = 0
            elif shortExit:  # 下跌过头平仓
                lastTrx = -1
                lastOpen = trxOpen
                lastGain = winpct
                lastTrxIdx = idx
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '下跌过头平仓'
                df.iloc[idx, df.columns.get_loc('reason')] = '跌过头'
                df.iloc[idx, df.columns.get_loc('action')] = -2
                df.iloc[idx, df.columns.get_loc('RET')] = -df['pctChg'].values[idx] - trade_rate
                df.iloc[idx, df.columns.get_loc('trxClose')] = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = 0
            elif df['flag'].values[idx] in [0, -1]:  # 空头继续持仓
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '空头持仓'
                df.iloc[idx, df.columns.get_loc('action')] = -1
                df.iloc[idx, df.columns.get_loc('RET')] = -df['pctChg'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
            elif df['flag'].values[idx] == 1:  # 空头转多头
                df.iloc[idx, df.columns.get_loc('winpct')] = winpct
                df.iloc[idx, df.columns.get_loc('winpctMax')] = winpctMax
                df.iloc[idx, df.columns.get_loc('remark')] = '空头转多头'
                df.iloc[idx, df.columns.get_loc('reason')] = '空转多'
                df.iloc[idx, df.columns.get_loc('action')] = 0
                df.iloc[idx, df.columns.get_loc('RET')] = -df['pctChg'].values[
                    idx] - 2 * trade_rate
                trxOpen = df['close'].values[idx]
                df.iloc[idx, df.columns.get_loc('trxOpen')] = trxOpen
                df.iloc[idx, df.columns.get_loc('trxClose')] = trxOpen
                df.iloc[idx, df.columns.get_loc('trxBars')] = idx - barOpen
                status = 1
                openReason = df['reason'].values[idx]
                barOpen = idx
        else:
            print('idx', idx, 'flag', df['flag'].values[idx], 'status', status)

    df['curve'] = (df['RET'] + 1).cumprod()
    return df[['trxOpen','trxClose', 'action', 'RET', 'curve','winpct', 'winpctMax', 'trxBars', 'remark', 'reason']]


def drawAllCCBs2min5B():
    global backset, gaphist_pct,bins,png_dict

    # try:
    #     png_dict = getMyOptions()
    # except:
    #     png_dict = {}

    # backset = 0
    trade_rate = 1/10000 # 手续费

    # etf_color_dict = {'上证50':'darkorange', '沪深300':'green','中证500':'royalblue','创业板50':'lightcoral'}

    periodkey = '5分钟K线'
    period = kline_dict[periodkey]
    klines=kline_qty[periodkey]
    ma_para=kline_mapara[periodkey]

    etf_5m_threshold = {'上证50': 0.1582, '沪深300': 0.1705, '中证500': 0.2039, '创业板50': 0.2649}


    df_all = pd.DataFrame()

    for k,v in etf_dict.items():
        df_single = getSingleCCBData(k,period,backset, klines,ma_para)

        # df_single[['pctChg', 'ccbma5', 'ccbma20', 'cm5', 'cm20', 'trxOpen','trxClose','cm20upCnt5','cm20up5cnt20','cm20dw5cnt20',
        #            'flag', 'action', 'RET', 'curve','winpct',  'trxBars', 'remark', 'reason']] = myStrategy(df_single,
        #                     cutloss=-0.005, cutprofit=0.01)
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
        gap_threshold = etf_5m_threshold[k]
        df_single.loc[(df_single['gap']>gap_threshold),'gapSig'] = df_single['gap']
        df_single.loc[(df_single['gap']<-1*gap_threshold),'gapSig'] = df_single['gap']

        df_single['ccbma5'] = df_single['ccb'].rolling(5).mean()
        df_single['ccbma20'] = df_single['ccb'].rolling(20).mean()
        df_single['ccbmgap'] = (df_single['ccbma5'] - df_single['ccbma20'])/df_single['ccbma5']

        df_single['ccbhhv30'] = df_single['ccbh'].rolling(30).max()
        df_single['ccbllv30'] = df_single['ccbl'].rolling(30).min()
        df_single['ccbcp30'] = df_single.apply(lambda x: (x['ccb']-x['ccbllv30'])/(x['ccbhhv30']-x['ccbllv30']), axis=1)
        df_single['ccbgap'] = df_single['ccp30']-df_single['ccbcp30']
        df_single['ccbgapm10'] = df_single['ccbgap'].rolling(10).mean()

        df_single.loc[(df_single['ccbgap']>df_single['ccbgapm10']) & (df_single['ccbgap'].shift(1)<df_single['ccbgapm10'].shift(1)),'up'] = -1.5  # (df_single['mark']>=0) &
        df_single.loc[(df_single['ccbgap']<df_single['ccbgapm10']) & (df_single['ccbgap'].shift(1)>df_single['ccbgapm10'].shift(1)),'dw'] = -1.5

        df_single.reset_index(drop=True,inplace=True)

        firstbar = df_single['time'].tolist().index('15:00')
        df_single = df_single[firstbar:]
        # df_single.reset_index(drop=True, inplace=True)
        # df_single.reset_index(drop=False,inplace=True)
        df_single['etf'] = k
        df_all = pd.concat([df_all, df_single])


    df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'ccbh', 'ccbl','ccbo', 'close', 'high', 'low','open',
                               'pctChg', 'ccbma5', 'ccbma20', 'cm5', 'cm20', 'ccbgap','ccbgapm10','up','dw'], dropna=False)#aggfunc=original_strings, fill_value='')

    # df_labels = df_all.pivot_table(index='datetime',columns='etf',values=['reason'], dropna=False,aggfunc=original_strings, fill_value='')
    # df_pivot = pd.merge(df_pivot, df_labels, left_index=True, right_index=True)

    df_pivot.reset_index(drop=False,inplace=True)
    df_pivot.reset_index(drop=False,inplace=True)
    lastBar = df_pivot[('datetime','')].values[-1][5:].replace('-','').replace(':','').replace(' ','_')

    fig, ax = plt.subplots(4, 1, figsize=(10*1,9), sharex=True)
    # plt.suptitle('ETF期权持仓比  时间戳:' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))

    # else:
    tickGap = 24
    xlables = [i[5:].replace('-','').replace(':','').replace(' ','_') for i in df_pivot[( 'datetime','')][::tickGap]]
    plt.xticks(range(len(df_pivot))[::tickGap],xlables,rotation=90)

    for x,k in zip(ax,  etf_dict.keys()):
    # for x,k in zip([ax],  ['上证50']):

        dftemp = df_pivot[[('index', ''), ('open', k), ('close', k), ('high', k), ('low', k)]]
        dftemp.columns = ['index', 'open', 'close', 'high', 'low']
        line_seg1, line_seg2, line_seg3 = construct_ohlc_collections(dftemp, wid=0.4)
        x.add_collection(line_seg1)
        x.add_collection(line_seg2)
        x.add_collection(line_seg3)

        # x.plot(df_pivot.index, df_pivot[('close',k)], label=k, linewidth=1, color='red',marker='.', markersize=4,alpha=0.7)
        x.plot(df_pivot.index, df_pivot[('cm5', k)], label='ma5', linewidth=1, linestyle='dotted', color='red', alpha=1.)
        # x.plot(df_pivot.index, df_pivot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='-.', color='red', alpha=1.)

        x2 = x.twinx()
        # x2.plot(df_pivot.index, df_pivot[('ccb',k)], label='ccb', linewidth=0.8, color='green')
        df_tmp = df_pivot[[('ccb', k), ('ccbh', k), ('ccbl', k), ('ccbo', k)]].copy()
        df_tmp.columns = ['close', 'high', 'low', 'open']
        ccbline_seg1, ccbline_seg2, ccbbar_segments = getKlineObjects(df_tmp, linewidths=1.2, bar_width=0.2)
        x2.add_collection(ccbline_seg1)
        x2.add_collection(ccbline_seg2)
        x2.add_collection(ccbbar_segments)
        x2.plot(df_pivot.index, df_pivot[('ccbma5', k)], label='ma5', linewidth=0.9, linestyle='dotted', color='green')
        x2.plot(df_pivot.index, df_pivot[('ccbma20', k)], label='ma20', linewidth=0.6, linestyle='-.', color='green')
        # x2.plot(df_pivot.index, df_pivot[('ccbRenkoM10', k)], linewidth=0.5, linestyle='-', color='blue')
        # yl, yh = x2.get_ylim()
        # for index, row in df_pivot.iterrows():
        #     x2.axvline(x=index, ymin=(row[('ccbl',k)]-yl)/(yh-yl), ymax=(row[('ccbh',k)]-yl)/(yh-yl), color='blue',linewidth=0.4)#linestyles='dashed',
        # x2.vlines(df_pivot.index, df_pivot[('ccbl',k)], df_pivot[('ccbh',k)], colors='blue', linewidth=0.4,alpha=1.)
        # x2.fill_between(df_pivot.index, df_pivot[('ccbl', k)], df_pivot[('ccbh', k)], color='blue', zorder=-26, alpha=0.2)

        # ax2.plot(df_single.index, df_single['ccbm5_ma'], label='ccbm5ma', linewidth=0.5, linestyle='dotted',color='blue')
        # x2.plot(df_pivot.index, df_pivot[('ccbm30_ma',k)], label='ccbm30ma', linewidth=0.5, linestyle='-.',color='blue')
        x2.set_yticks([])

        x3 = x.twinx()
        x3.plot(df_pivot.index, df_pivot[('ccbgap', k)], color='blue', linewidth=0.7, linestyle='-')  # marker='.',
        x3.plot(df_pivot.index, df_pivot[('ccbgapm10', k)], color='blue', linestyle='--', linewidth=0.5)
        # x3.set_yticks([])
        x3.scatter(df_pivot.index, df_pivot[('up', k)], marker='^', s=64, c='red',alpha=0.7)
        x3.scatter(df_pivot.index, df_pivot[('dw', k)], marker='v',s=64, c='green',alpha=0.7)
        x3.set_ylim(-2, 2)
        # x3.hlines(0,xmin=df_pivot.index.min(), xmax=df_pivot.index.max(),color='green')
        # x3.scatter(df_pivot.index,df_pivot[('gapdw',k)], s=4, c='g',marker='.',alpha=0.8)

        # x4 = x.twinx()
        # x4.plot(df_pivot.index,df_pivot[('cm20upCnt5',k)],label='cm20upCnt5', linewidth=3, color='red',alpha=0.25,zorder=-10)
        # # x4.plot(df_pivot.index,df_pivot[('cm20up5cnt20',k)],label='cm20up5cnt20', linewidth=2, color='red',alpha=0.3,zorder=-10)
        # # x4.plot(df_pivot.index,df_pivot[('cm20dw5cnt20',k)],label='cm20dw5cnt20', linewidth=2, color='green',alpha=0.3,zorder=-10)
        # # x4.hlines(0, xmin=df_pivot.index.min(), xmax=df_pivot.index.max(), color='black')
        # x4.set_ylim(-0.5, 5.5)
        # x4.set_yticks([])

        # x5 = x.twinx()
        # x5.plot(df_pivot.index,df_pivot[('score',k)], linewidth=0.5, color='violet',zorder=-10)
        # # x5.set_ylim(-3, 3)
        # x5.set_yticks([])

        # x.legend(loc='upper left')
        # x2.legend(loc='center left')
        # x3.legend(loc='lower left')

        x.minorticks_on()
        x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.1)

        if k in png_dict.keys():
            x.text(0.25,0.9,  png_dict[k], horizontalalignment='center',transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.suptitle(f'{df_pivot[("datetime","")].values[-1][-11:]}',x=0.6, y=0.98)

    plt.savefig('output\\持续监控ETF期权5分钟监控_v2.0.png')
    fig.clf()
    plt.close(fig)

def main():

    global api, Exapi, factor, dayr1,png_dict

    if (time.strftime("%H%M", time.localtime()) > '0900' and time.strftime("%H%M", time.localtime()) <= '0930'):
        logger.info('waiting market, sleep 40s')
        time.sleep(40)

    try:
        api.close()
        Exapi.close()
    except:
        time.sleep(5)

    try:
        api = TdxHq_API(heartbeat=True)
        Exapi = TdxExHq_API(heartbeat=True)

        if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport )==False:
            logger.error('connection to TDX server not available')
        if TestConnection(Exapi, 'ExHQ', conf.ExHQsvr, conf.ExHQsvrport )==False:
            logger.error('connection to Ex TDX server not available')

        while (time.strftime("%H%M", time.localtime())>='0930' and time.strftime("%H%M", time.localtime())<='1502'):

            if (time.strftime("%H%M", time.localtime())>'1130' and time.strftime("%H%M", time.localtime())<'1300'):
                logger.info('sleep 60s')
                time.sleep(60)
            else:
                try:
                    png_dict = getMyOptions()
                except:
                    png_dict = {}
                drawAllCCBs2min5B()
                drawAllCCBmin1A()
                drawAllCCBmin1C()
                # drawAllCCBmin1B()
                plotAllzjlx()
                time.sleep(20)

        drawAllCCBs2min5B()
        drawAllCCBmin1A()
        drawAllCCBmin1C()
        # drawAllCCBmin1B()
        plotAllzjlx()

        api.close()
        Exapi.close()
        return
    except Exception as e: # work on python 3.x
        logger.error('exception msg: '+ str(e))
        logger.error(' *****  exception, restart main ***** ')
        time.sleep(5)
        main()


if __name__ == '__main__':

    prog_start = time.time()
    logger.info('-------------------------------------------')
    logger.info('Job start !!! ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))

    etf_ccb_dict = {'上证50':'UC050', '沪深300':'UC300','中证500':'UC500','创业板50':'UC915'}
    etf_dict = {'上证50':'zs000016','沪深300':'zs000300','中证500':'zs000905','创业板50':'399673'}
    etf_dict2 = {'上证50':'510050','沪深300':'510300','中证500':'510500','创业板50':'159915'}
    etfcode_dict = {'上证50':'510050','沪深300':'510300','中证500':'510500','创业板50':'159915'}

    kline_dict = {"5分钟K线": 0, "15分钟K线": 1, "30分钟K线": 2, "60分钟K线": 3, "日K线bb": 4, "周K线": 5,
                  "月K线": 6, "1分钟bb": 7, "1分钟K线": 8, "日K线": 9, "季K线": 10, "年K线": 11}
    kline_qty = {"5分钟K线": 48*2+1, "15分钟K线": 1001, "30分钟K线": 0, "60分钟K线": 601, "日K线bb": 301, "周K线": 71,
                 "月K线": 60, "1分钟bb": 1001, "1分钟K线": 501, "日K线": 301, "季K线": 31, "年K线": 11}
    kline_mapara = {"5分钟K线": 150, "15分钟K线": 50, "30分钟K线": 50, "60分钟K线": 14, "日K线bb": 3, "周K线": 3,
                    "月K线": 2, "1分钟bb": 1001, "1分钟K线": 1001, "日K线": 20, "季K线": 3, "年K线": 1}

    backset = 0
    trade_rate = 0/10000 # 手续费
    dte_low = 5
    dte_high = 45
    close_Threshold = 0.090

    threshold_pct = gaphist_pct = 75
    bins = 100

    cutloss = -0.005
    cutprofit = 0.01
    trendKline = 240

    png_dict = {}
    # etf_dict = {'上证50':'zs000016','沪深300':'zs000300','中证500':'zs000905','创业板50':'399673'}
    # etf_ccb_dict = {'上证50':'UC050', '沪深300':'UC300','中证500':'UC500','创业板50':'UC915'}
    # etf_color_dict = {'上证50':'darkorange', '沪深300':'green','中证500':'royalblue','创业板50':'lightcoral'}

    api = TdxHq_API(heartbeat=True)
    if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport) == False: # or \
        logger.error('connection to TDX server not available')

    Exapi = TdxExHq_API(heartbeat=True)
    if TestConnection(Exapi, 'ExHQ', conf.ExHQsvr, conf.ExHQsvrport )==False:
        logger.error('connection to EXHQ server not available')

    ttt = pd.DataFrame(etfData(api, Exapi,'510050', '上证50', 'BK0611','UC050',backset, 240, 9).api.get_security_bars(8,1, '510050', 0, 240))
    today = ttt['datetime'].values[-1][:10]

    try:
        png_dict = getMyOptions()
    except:
        png_dict = {}

    factor = calAmtFactor(5)
    factor = factor+[1.00]

    main()

    # drawAllCCBmin1A()
    # drawAllCCBs2min5B()
    # drawAllCCBmin1C()
    # plotAllzjlx()
    # drawAllCCBs2min5B()

    api.close()
    Exapi.close()

    time_end = time.time()
    logger.info('-------------------------------------------')
    logger.info(f'Job completed!!!  All time costed: {(time_end - prog_start):.0f}秒')

