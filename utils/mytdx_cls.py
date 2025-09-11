from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
import pandas as pd
import requests, re,json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

'''
[tdx_hosts]
host_1 = 招商深圳云,39.108.28.83,7709
host_2 = 招商深圳云2,109.244.69.145, 7709
host_3 = 招商北京云,39.105.251.234, 7709
host_4 = 招商北京云2,120.53.204.206, 7709
host_5 = 招商北京云3,109.244.7.166, 7709
host_6 = 招商广州云,111.230.189.225, 7709
host_7 = 招商广州云2,106.53.111.126, 7709
host_8 = 招商上海云,47.100.132.162, 7709
host_9 = 招商上海云2,43.145.21.43, 7709
host_10 = 招商教育网,116.57.224.5, 7709
host_11 = 招商深圳电信,183.62.101.52, 7709
host_12 = 招商深圳联通,58.251.16.180, 7709

[tdx_exhosts]
exhost_1 = 招商深圳云1,39.108.28.120,7727
exhost_2 = 招商广州云,134.175.214.53, 7727
exhost_3 = 招商北京云,182.92.255.107, 7727
exhost_4 = 招商北京云2,140.143.179.226, 7727
exhost_5 = 招商上海云,47.100.132.106, 7727
exhost_6 = 招商上海云2,212.64.116.126, 7727
'''

class mytdxData(object):

    def __init__(self, hq_hosts, Exhq_hosts, speed_test=False):
        api = TdxHq_API(heartbeat=True)
        Exapi = TdxExHq_API(heartbeat=True)
        api_connected = False
        Exapi_connected = False
        self.hq_hosts = hq_hosts
        self.Exhq_hosts = Exhq_hosts

        if speed_test==False:
            api = TdxHq_API(heartbeat=True)
            apilist = pd.DataFrame(hq_hosts)
            apilist.columns = ['name', 'ip', 'port']
            if self.TestConnection(api, 'HQ', apilist['ip'].values[0], apilist['port'].values[0]) == False:  # or \
                if self.TestConnection(api, 'HQ', apilist['ip'].values[1], apilist['port'].values[1]) == False:  # or \
                    if self.TestConnection(api, 'HQ', apilist['ip'].values[2],
                                           apilist['port'].values[2]) == False:  # or \
                        if self.TestConnection(api, 'HQ', apilist['ip'].values[3],
                                               apilist['port'].values[3]) == False:  # or \
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
            exapilist = pd.DataFrame(Exhq_hosts)
            exapilist.columns = ['name', 'ip', 'port']
            if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[0],
                                   exapilist['port'].values[0]) == False:  # or \
                if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[1],
                                       exapilist['port'].values[1]) == False:  # or \
                    if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[2],
                                           exapilist['port'].values[2]) == False:  # or \
                        if self.TestConnection(Exapi, 'ExHQ', exapilist['ip'].values[3],
                                               exapilist['port'].values[3]) == False:  # or \
                            print('All ExHQ server Failed!!!')
                        else:
                            print(f'connection to ExHQ server[3]!{exapilist["name"].values[3]}')
                    else:
                        print(f'connection to ExHQ server[2]!{exapilist["name"].values[2]}')
                else:
                    print(f'connection to ExHQ server[1]!{exapilist["name"].values[1]}')
            else:
                print(f'connection to ExHQ server[0]!{exapilist["name"].values[0]}')

        else:
            # 测速并排序HQ服务器
            sorted_hq_hosts = self.speed_test_hosts(hq_hosts, 'HQ')
            for i in range(len(sorted_hq_hosts)):
                name = sorted_hq_hosts[i][0]
                ip = sorted_hq_hosts[i][1]
                port = sorted_hq_hosts[i][2]
                if self.TestConnection(api, 'HQ', ip, port) == False:
                    continue
                else:
                    api_connected = True
                    print(f'connection to HQ server[{i}]!{name}{ip} (response time: {sorted_hq_hosts[i][3]:.3f}s)')
                    break
            if api_connected == False:
                print('All HQ server Failed!!!')

            # 测速并排序ExHQ服务器
            sorted_exhq_hosts = self.speed_test_hosts(Exhq_hosts, 'ExHQ')
            for i in range(len(sorted_exhq_hosts)):
                name = sorted_exhq_hosts[i][0]
                ip = sorted_exhq_hosts[i][1]
                port = sorted_exhq_hosts[i][2]
                if self.TestConnection(Exapi, 'ExHQ', ip, port) == False:
                    continue
                else:
                    Exapi_connected = True
                    print(f'connection to ExHQ server[{i}]!{name}{ip} (response time: {sorted_exhq_hosts[i][3]:.3f}s)')
                    break
            if Exapi_connected == False:
                print('All ExHQ server Failed!!!')

        self.api = api
        self.Exapi = Exapi
        self.useless_cols = ['year', 'month', 'day', 'hour', 'minute', 'preclose', 'change']
        self.period_dict = {"5min": 0, "15min": 1, "30min": 2, "60min": 3, "week": 5,
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

    def close(self):
        self.api.close()
        self.Exapi.close()

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
            res = requests.get(url, proxies={})
            data1 = pd.DataFrame(json.loads(res.text)['result']['data'])
            if len(data1)==0:
                return pd.DataFrame()
            else:
                data1.rename(columns={'EX_DIVIDEND_DATE':'date','SECURITY_CODE':'code','REPORT_TYPE':'type','PLAN_EXPLAIN':'deal'}, inplace=True)
                data1 = data1[data1['type'].str.contains('分配')]
                data1['date'] = data1['date'].apply(lambda x: x.replace('/','-'))
                # data1['deal'] = data1['deal'].apply(lambda x: float(re.findall(r'\d+\.\d+(?=[^.\d]*$)',x)[-1]))
                data1['deal'] = data1['deal'].apply(lambda x: float(re.findall(r'(\d+\.?\d*)', x)[-1]))
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

        if code in ['VHSI','HSAHP','HSREIT','HSU','HSCAHSI','HSP','HSCICS',
                    'HSSCCS','HSCIEN','HSSCCSI','HSCITC','HSCIUT','HSI']:  # 香港指数
            mkt = 27
            isIndex = True
            fuquan = False
            return mkt, code, fuquan, isIndex

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
            elif code[:2] in ['51','52','58','56','60','68','50','88','11','99']:
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
                df_k.reset_index(drop=True, inplace=True)
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
                            if row['date']<=df_k['date'].values[0] or row['date']>=df_k['date'].values[-1]:
                                continue
                            else:
                                pass
                                ##  bug 159922 @ 20241129扩股 20241202除权除息 导致复权后价格变化异常 ---  待修改
                                # idx = df_k[df_k['date']>=row['date']].index[0]
                                # predate = df_k.loc[idx-1, 'date']
                                # thisdate = df_k.loc[idx, 'date']
                                # if row['name'] == '扩缩股':
                                #     preclose = df_k.loc[df_k['date'] == predate, 'preclose'].values[0]
                                #     thisclose = df_k.loc[df_k['date'] == thisdate, 'close'].values[0]
                                #     change_new = (thisclose * row['suogu']) / preclose - 1
                                #     df_k.loc[df_k['date'] == row['date'], 'change'] = change_new
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

                    df_k.iloc[0,df_k.columns.get_loc('change')] = 0.0
                    df_k[['open', 'close', 'high', 'low']] = self.cal_right_price(df_k, type='前复权')

                    for column in self.useless_cols:
                        if column in df_k.columns:
                            df_k = df_k.drop(columns=column)

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

        elif mkt in [71,27]:  # 港股通71  香港沪通27
            if qty<=600:
                df_k = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 0 + backset, qty))
                if len(df_k)==0:
                    return pd.DataFrame()
                df_k['date'] = df_k['datetime'].apply(lambda x: x[:10])
            else:
                df_k = pd.DataFrame()
                for i in range(qty//600):
                    temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*i+backset, 600))
                    df_k = pd.concat([temp,df_k ])
                temp = pd.DataFrame(self.Exapi.get_instrument_bars(period, mkt, code, 600*(qty//600)+backset, qty%600))
                df_k = pd.concat([temp, df_k])
                df_k['date'] = df_k['datetime'].apply(lambda x: x[:10])

            if fuquan : # 港股通复权
                df_k.reset_index(drop=True, inplace=True)
                df_k['preclose'] = df_k['close'].shift(1)
                df_k['change'] = df_k['close'] / df_k['preclose'] - 1
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
                df_k.iloc[0, df_k.columns.get_loc('change')] = 0.0
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
        elif mkt in [68] and isIndex==False:
            data = pd.DataFrame(self.Exapi.get_history_minute_time_data(mkt,code,day))
        else:
            data = pd.DataFrame()
        return data

    def get_kline_data(self,code, backset=0, klines=200, period=9):
        df=self.fuquan202409(code, backset, klines, period)

        if len(df)==0:
            return pd.DataFrame()

        if '成交额' not in df.columns:
            df.rename(columns={'amount': '成交额'}, inplace=True)
            if 'vol' in df.columns:
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

    def  get_transaction_data(self, code, backset=0, qty=200, dateint=0):

        # 参数：市场代码， 股票代码，起始位置， 数量
        # 如： 0, 000001, 0, 10
        # api.get_transaction_data(TDXParams.MARKET_SZ, '000001', 0, 30)

        # 参数：市场代码， 股票代码，起始位置，日期
        # 数量
        # 如： 0, 000001, 0, 10, 20170209
        # api.get_history_transaction_data(TDXParams.MARKET_SZ, '000001', 0, 10, 20170209)

        mkt,code,fuquan,isIndex = self.get_market_code(code)

        if dateint!=0:
            if qty <= 1800:
                df = pd.DataFrame(self.api.get_history_transaction_data(mkt, code, backset, qty, dateint))
            else:
                df = pd.DataFrame()
                for i in range(qty // 1800):
                    temp = pd.DataFrame(self.api.get_history_transaction_data(mkt, code, 1800*i+backset, 1800, dateint))
                    df = pd.concat([temp, df])
                temp = pd.DataFrame(self.api.get_history_transaction_data(mkt, code, 1800*(qty//1800)+backset, qty%1800, dateint))
                df = pd.concat([temp, df])
        else:
            if qty <= 1800:
                df = pd.DataFrame(self.api.get_transaction_data(mkt, code, backset, qty))
            else:
                df = pd.DataFrame()
                for i in range(qty // 1800):
                    temp = pd.DataFrame(self.api.get_transaction_data(mkt, code, 1800*i+backset, 1800))
                    df = pd.concat([temp, df])
                temp = pd.DataFrame(self.api.get_transaction_data(mkt, code, 1800*(qty//1800)+backset, qty%1800))
                df = pd.concat([temp, df])
        df.reset_index(drop=True, inplace=True)
        return df

    def get_close_list(self, stklist,backtest=0):
        result = pd.DataFrame()

        for stk in stklist:
            tmp = self.fuquan202409(stk, backtest, qty=2, period=9)
            if len(tmp)==0:
                continue
            result = result.append({'code': stk, 'close': tmp['close'].values[-1]}, ignore_index=True)

        return result

    def speed_test_host(self, host_info, api_type):
        """
        测试单个服务器的响应时间
        """
        name, ip, port = host_info[:3]
        api = TdxHq_API() if api_type == 'HQ' else TdxExHq_API()

        try:
            start_time = time.time()
            is_connect = api.connect(ip, port)
            connect_time = time.time() - start_time

            if is_connect:
                # 尝试获取简单数据以测试响应速度
                if api_type == 'HQ':
                    api.get_security_count(0)  # 获取市场证券数量
                else:
                    api.get_instrument_count()  # 获取合约数量
                response_time = time.time() - start_time
                api.close()
                return (name, ip, port, response_time)
            else:
                api.close()
                return (name, ip, port, float('inf'))
        except Exception as e:
            if 'api' in locals():
                api.close()
            return (name, ip, port, float('inf'))

    def speed_test_hosts(self, hosts, api_type, max_workers=5):
        """
        并行测试所有服务器速度并排序
        """
        print(f"Testing {api_type} servers speed...")
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有测试任务
            future_to_host = {
                executor.submit(self.speed_test_host, host, api_type): host
                for host in hosts
            }

            # 收集测试结果
            for future in as_completed(future_to_host):
                result = future.result()
                results.append(result)

        # 按响应时间排序，过滤掉连接失败的服务器
        results = [r for r in results if r[3] != float('inf')]
        results.sort(key=lambda x: x[3])

        # 输出测速结果
        print(f"\n{api_type} servers speed test results:")
        for i, (name, ip, port, response_time) in enumerate(results):
            if response_time == float('inf'):
                print(f"  {i + 1}. {name} ({ip}:{port}) - Connection failed")
            else:
                print(f"  {i + 1}. {name} ({ip}:{port}) - {response_time:.3f}s")

        return results

    def reconnect(self):
        """
        重写重连方法，加入测速功能
        """
        self.close()  # 关闭当前连接

        api = TdxHq_API(heartbeat=True)
        Exapi = TdxExHq_API(heartbeat=True)
        api_connected = False
        Exapi_connected = False

        # 测速并排序HQ服务器
        sorted_hq_hosts = self.speed_test_hosts(self.hq_hosts, 'HQ')
        for i in range(len(sorted_hq_hosts)):
            name = sorted_hq_hosts[i][0]
            ip = sorted_hq_hosts[i][1]
            port = sorted_hq_hosts[i][2]
            if self.TestConnection(api, 'HQ', ip, port) == False:
                continue
            else:
                api_connected = True
                print(f'reconnection to HQ server[{i}]!{name}{ip} (response time: {sorted_hq_hosts[i][3]:.3f}s)')
                break
        if not api_connected:
            print('All HQ server Failed!!!')

        # 测速并排序ExHQ服务器
        sorted_exhq_hosts = self.speed_test_hosts(self.Exhq_hosts, 'ExHQ')
        for i in range(len(sorted_exhq_hosts)):
            name = sorted_exhq_hosts[i][0]
            ip = sorted_exhq_hosts[i][1]
            port = sorted_exhq_hosts[i][2]
            if self.TestConnection(Exapi, 'ExHQ', ip, port) == False:
                continue
            else:
                Exapi_connected = True
                print(f'reconnection to ExHQ server[{i}]!{name}{ip} (response time: {sorted_exhq_hosts[i][3]:.3f}s)')
                break
        if not Exapi_connected:
            print('All ExHQ server Failed!!!')

        self.api = api
        self.Exapi = Exapi

if __name__ == '__main__':


    tdx_hosts = [('招商深圳云', '39.108.28.83', 7709), ('招商深圳云2', '109.244.69.145', 7709),
     ('招商北京云', '39.105.251.234', 7709), ('招商北京云2', '120.53.204.206', 7709),
     ('招商北京云3', '109.244.7.166', 7709), ('招商广州云', '111.230.189.225', 7709),
     ('招商广州云2', '106.53.111.126', 7709), ('招商上海云', '47.100.132.162', 7709),
     ('招商上海云2', '43.145.21.43', 7709), ('招商教育网', '116.57.224.5', 7709),
     ('招商深圳电信', '183.62.101.52', 7709), ('招商深圳联通', '58.251.16.180', 7709)]
    tdx_exhosts = [('招商深圳云1', '39.108.28.120', 7727), ('招商广州云', '134.175.214.53', 7727),
     ('招商北京云', '182.92.255.107', 7727), ('招商北京云2', '140.143.179.226', 7727),
     ('招商上海云', '47.100.132.106', 7727), ('招商上海云2', '212.64.116.126', 7727)]


    mytdx =mytdxData(tdx_hosts, tdx_exhosts)

    ttt = mytdx.get_close_list(['10008885','510300','510500'])
    df1 = mytdx.get_kline_data('000001', backset=0, klines=200, period=9)

    mytdx.reconnect()
    mytdx.close()

    print('test done!')