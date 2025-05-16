# data/etf_fetcher.py
import pandas as pd
import numpy as np
import requests, json, time
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed



class ETFDataFetcher:
    def __init__(self, tdx_data, etf_dict2: dict, ccb_dict: dict, bk_dict: dict, backset: int = 0):
        self.tdx_data = tdx_data
        self.etf_dict2 = etf_dict2
        self.backset = backset
        self.etf_ccb_dict = ccb_dict
        self.etf_bk_dict = bk_dict
        factor = self.calAmtFactor(5)
        self.factor = factor + [1.00]
        self.ccb_range = {}
        self.ccb_pctChg = {}
        for k,v in self.etf_ccb_dict.items():
            ccb_temp = self.tdx_data.get_kline_data(v, backset=0, klines=200, period=9)
            self.ccb_range[f'{k}_max'] = ccb_temp['close'].max()
            self.ccb_range[f'{k}_min'] = ccb_temp['close'].min()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def safe_get_request(self,url):
        return requests.get(url)

    def calAmtFactor(self,n):

        # df_day = pd.DataFrame(api.get_index_bars(9, 1, '999999', 0, n+1))
        df_day = self.tdx_data.get_kline_data('999999', 0, n + 1, 9)

        if 'date' not in df_day.columns:
            df_day['date'] = df_day['datetime'].apply(lambda x: x[:10])
        if 'amount' not in df_day.columns:
            df_day['amount'] = df_day['成交额']
        daylist = df_day['date'].values[:-1]

        times = (n + 1) * 240 // 800 + 1 if (n + 1) * 240 % 800 > 0 else (n + 1) * 240 // 800
        df_min = pd.DataFrame()
        for i in range(times - 1, -1, -1):
            # temp = pd.DataFrame(api.get_index_bars(8, 1, '999999', i*800, 800))
            temp = self.tdx_data.get_kline_data('999999', i * 800, 800, 8)
            if len(temp) > 0:
                df_min = pd.concat([df_min, temp])

        if 'amount' not in df_min.columns:
            df_min['amount'] = df_min['成交额']
        df_min['date'] = df_min['datetime'].apply(lambda x: x[:10])
        df_min['time'] = df_min['datetime'].apply(lambda x: x[11:])
        df_min = df_min[(df_min['date'] >= daylist[0]) & (df_min['date'] <= daylist[-1])]
        df_min.reset_index(drop=True, inplace=True)

        ttt = df_min.pivot_table(index='time', values='amount', aggfunc='sum')
        ttt['cum'] = ttt['amount'].cumsum()
        ttt['ratio'] = ttt['cum'].values[-1] / ttt['cum']

        return list(ttt['ratio'].values)

    def fetch_single_etf_data(self, name: str, period: int = 9, klines: int = 300) -> pd.DataFrame:
        code = self.etf_dict2[name]
        df_single = self.tdx_data.get_kline_data(code, backset=self.backset, klines=klines, period=period)
        if '成交额' in df_single.columns:
            df_single.rename(columns={'成交额': 'amount'}, inplace=True)
        df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00', '11:30'))
        return df_single

    def fetch_bk_zjlx_rt(self, bkcode: str) -> pd.DataFrame:
        url = f'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=90.{bkcode}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56&ut=fa5fd1943c7b386f172d6893dbfba10b&cb=jQuery112406142175621622367_1615545163205&_={int(time.time())}'
        # res = requests.get(url)
        res = self.safe_get_request(url)
        try:
            data1 = json.loads(res.text[42:-2])['data']['klines']
        except Exception:
            return pd.DataFrame({'datetime': ['2000-01-01'], 'boss': [0]})
        min_df = pd.DataFrame([i.split(',') for i in data1], columns=['datetime', 'boss', 'small', 'med', 'big', 'huge'])
        min_df = min_df[['datetime', 'boss']]
        min_df['boss'] = min_df['boss'].astype(float) / 1e8
        return min_df

    def get_BK_Zjlx(self,bkcode):

        url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=90.' + bkcode + \
              '&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56&ut=fa5fd1943c7b386f172d6893dbfba10b&cb=jQuery112406142175621622367_1615545163205&_=1615545163206'
        # res = requests.get(url)
        res = self.safe_get_request(url)

        try:
            data1 = json.loads(res.text[42:-2])['data']['klines']
        except:
            return pd.DataFrame({'datetime': ['2000-01-01'], 'boss': [0]})
        min = pd.DataFrame([i.split(',') for i in data1], columns=['datetime', 'boss', 'small', 'med', 'big', 'huge'])
        min.drop(labels=['small', 'med', 'big', 'huge'], axis=1, inplace=True)
        # min['time'] = min['time'].astype('datetime64[ns]')
        min['boss'] = min['boss'].astype('float') / 100000000

        return min
    def get_Single_CCB_Data(self, name, backset=0, klines=500, period=8):

        code = self.etf_dict2[name]
        df_single= self.tdx_data.get_kline_data(code, backset=backset, klines=klines, period=period)
        if '成交额' in df_single.columns:
            df_single.rename(columns={'成交额':'amount'},inplace=True)
        df_single.reset_index(drop=True,inplace=True)
        if len(df_single)==0:
            print(f'getSingleCCBData {code} kline error,quitting')
            return
        # df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30') if x[-5:]=='13:00' else x)
        df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30'))

        ccbcode = self.etf_ccb_dict[name]
        df_ccb =  self.tdx_data.get_kline_data(ccbcode, backset=backset, klines=klines, period=period)

        if len(df_ccb)==0:
            print('getSingleCCBData {code} ccb error, quitting')
            return
        tmp = df_ccb[df_ccb['datetime'].str.contains('15:00')]
        if '15:00' in df_ccb['datetime'].values[-1]:
            preidx = tmp.index[-2]
        else:
            preidx = tmp.index[-1]
        pre_ccb = df_ccb.close.iloc[preidx]
        ccb_open = df_ccb.close.iloc[preidx+1]
        df_ccb.rename(columns={'close':'ccb','high':'ccbh','low':'ccbl','open':'ccbo'},inplace=True)
        df_ccb['ccbm20'] = df_ccb['ccb'].rolling(20).mean()
        df_ccb['ccbm20up'] = (df_ccb['ccbm20']>df_ccb['ccbm20'].shift(1)).astype('int')
        df_ccb['pre_ccb'] = pre_ccb
        df_ccb['ccb_open'] = ccb_open
        df_ccb['ccb_pct'] = df_ccb['ccb'].apply(lambda x: x/pre_ccb*1-1)
        df_ccb['ccb_open2pct'] =  df_ccb['ccb'].apply(lambda x: x/ccb_open*1-1)
        df_ccb['ccb_above_open'] =  df_ccb['ccb'].apply(lambda x: 1 if x>ccb_open else 0)
        self.ccb_pctChg[name] = df_ccb['ccb_pct'].values[-1]
        data = pd.merge(df_ccb[['datetime','ccb','ccbh','ccbl','ccbo','ccbm20','ccb_pct','pre_ccb','ccb_open','ccb_open2pct','ccb_above_open','ccbm20up']],
                        df_single[['datetime','open','close','high','low','volume','amount']], on='datetime',how='left')


        return data

    def MINgetZjlxDP(self):
        url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=1.000001&secid2=0.399001&' + \
              'fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&' + \
              'ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery18308174687833149541_1607783437004&_=1607783437202'
        res = self.safe_get_request(url)

        try:
            data1 = json.loads(res.text[41:-2])['data']['klines']
        except:
            return pd.DataFrame()
        min = pd.DataFrame([i.split(',') for i in data1], columns=['datetime', 'boss', 'small', 'med', 'big', 'huge'])
        min.drop(labels=['small', 'med', 'big', 'huge'], axis=1, inplace=True)
        # min['datetime'] = min['datetime'].astype('datetime64[ns]')
        min['boss'] = min['boss'].astype('float')
        min['net'] = min['boss'] - min['boss'].shift(1)
        min['bossma5'] = min['boss'].rolling(5).mean()

        return min

    def MINgetDPindex(self):

        # day_sh =  pd.DataFrame(api.get_index_bars(9, 1, '999999', 0, 30))
        day_sh = self.tdx_data.get_kline_data('999999', 0, 30, 9)
        lastday_amount_sh = day_sh['成交额'].values[-2]
        lastday_amount_sz = self.tdx_data.get_kline_data('399001', 0, 30, 9)['成交额'].values[-2]
        lastday_amount = (lastday_amount_sh + lastday_amount_sz) / 100000000

        if 'amount' not in day_sh.columns:
            day_sh['amount'] = day_sh['成交额']
        if 'vol' not in day_sh.columns:
            day_sh['vol'] = day_sh['volume']
        datelast = day_sh['datetime'].values[-2]
        preclose = day_sh[day_sh['datetime'] == datelast]['close'].values[-1]

        # df_sh =  pd.DataFrame(api.get_index_bars(8, 1, '999999', 0, 300))
        df_sh = self.tdx_data.get_kline_data('999999', 0, 300, 8)
        if 'amount' not in df_sh.columns:
            df_sh['amount'] = df_sh['成交额']
        if 'vol' not in df_sh.columns:
            df_sh['vol'] = df_sh['volume']

        df_sh['date'] = df_sh['datetime'].apply(lambda x: x[:10])
        df_sh['time'] = df_sh['datetime'].apply(lambda x: x[11:])
        df_sh = df_sh[(df_sh['datetime'] > datelast)]
        df_sh.reset_index(drop=True, inplace=True)

        # df_sz =  pd.DataFrame(api.get_index_bars(8, 0, '399001', 0, 300))
        df_sz = self.tdx_data.get_kline_data('399001', 0, 300, 8)
        if 'amount' not in df_sz.columns:
            df_sz['amount'] = df_sz['成交额']
        df_sz['date'] = df_sz['datetime'].apply(lambda x: x[:10])
        df_sz['time'] = df_sz['datetime'].apply(lambda x: x[11:])
        df_sz = df_sz[(df_sz['datetime'] > datelast)]
        df_sz.reset_index(drop=True, inplace=True)
        df_sz.rename(columns={'amount': 'amountsz'}, inplace=True)

        data = pd.merge(df_sh[['datetime', 'amount', 'vol', 'close', 'high', 'low']], df_sz[['datetime', 'amountsz']],
                        how='inner')
        data['allamt'] = data['amount'] + data['amountsz']
        data['amt'] = data['close'] * data['vol']
        data['amtcum'] = data['amt'].cumsum()
        data['volcum'] = data['vol'].cumsum()
        data['avg'] = data['amtcum'] / data['volcum']
        data['amt'] = data['amount'] + data['amountsz']
        data['amtcum'] = data['amt'].cumsum()
        data['factor'] = self.factor[:len(data)]
        data['amttrend'] = data['factor'] * data['amtcum']
        # data['amttrend'] = data['amttrend'].ffill()
        # data.iloc[-1, data.columns.get_loc('amttrend')] = np.nan

        data.drop(columns=['amtcum', 'volcum', 'amt', 'factor'], inplace=True)

        return data, preclose
    def get_DP_data(self):

        dp_zjlx = self.MINgetZjlxDP()

        dp_zjlx['datetime'] = dp_zjlx['datetime'].apply(lambda x: x.replace('13:00', '11:30'))

        # datetime,boss,net,bossma5
        dp_index, dp_preclose = self.MINgetDPindex()  # ['datetime', 'amount', 'vol', 'close', 'allamt', 'avg', 'amttrend']
        dp_index['datetime'] = dp_index['datetime'].apply(lambda x: x.replace('13:00', '11:30'))

        df_dp = pd.merge(dp_index, dp_zjlx, on='datetime', how='left')

        return df_dp, dp_preclose

    def get_ETF_data(self):

        # periodkey = '1分钟k线'
        # period = 8
        # klines= 500

        df_all = pd.DataFrame()

        for k,v in self.etf_dict2.items():
            df_single = self.get_Single_CCB_Data(k, backset=0, klines=500, period=8)
            self.ccb_range[f'{k}_CP'] =  (df_single['ccb'].values[-1]- self.ccb_range[f'{k}_min'])/(self.ccb_range[f'{k}_max']- self.ccb_range[f'{k}_min'])
            df_BKzjlx = self.get_BK_Zjlx(self.etf_bk_dict[k])
            df_single = pd.merge(df_single, df_BKzjlx[['datetime','boss']], on='datetime',how='left')

            tmp = df_single[df_single['datetime'].str.contains('15:00')]
            if '15:00' in df_single['datetime'].values[-1]:
                preidx = tmp.index[-2]
            else:
                preidx = tmp.index[-1]
            preclose =   df_single.loc[preidx,'close']
            df_single['preclose'] = preclose

            k_h = max(preclose, df_single['high'][preidx+1:].max())
            k_l = min(preclose, df_single['low'][preidx+1:].min())
            k_hh = k_l + (k_h - k_l) * 7.5 / 8
            k_ll = k_l + (k_h - k_l) * 0.5 / 8
            df_single['crossdw'] = np.nan
            df_single['crossup'] = np.nan
            df_single.loc[(df_single['close'] < k_hh) & (df_single['close'].shift(1) >= k_hh), 'crossdw'] = -0.5
            df_single.loc[(df_single['close'] > k_ll) & (df_single['close'].shift(1) <= k_ll), 'crossup'] = -0.5


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
            df_single['cm20up'] = (df_single['cm20']> df_single['cm20'].shift(1)).astype('int')

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

            df_single['sig'] = df_single[['pivotup', 'pivotdw', 'crossup', 'crossdw']].notnull().any(axis=1)
            df_single['sig'] = df_single.apply(lambda x: np.nan if x.sig == False else x.close, axis=1)
            df_single['sig'] = df_single['sig'].ffill()
            df_single['up'] = df_single.apply(lambda x: 0 if x.close > x.sig else np.nan, axis=1)
            df_single['dw'] = df_single.apply(lambda x: 0 if x.close < x.sig else np.nan, axis=1)


            df_single['etf'] = k
            df_all = pd.concat([df_all, df_single])
        df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'ccbm20','ccbm20up','ccb_pct','pre_ccb','ccb_open','ccb_open2pct','ccb_above_open',
                    'close', 'high', 'low','open','preclose','volume', 'cm5','cm20','cm20up',
                    'ccp60','pivotup','pivotdw','crossup','crossdw','up','dw','bosssigup','bosssigdw','boss','amount'], dropna=False)

        return df_pivot

    def get_full_data(self):


        df_dapan,dp_preclose = self.get_DP_data()
        self.dp_preclose = dp_preclose
        dp_h = max(dp_preclose, df_dapan.close.max())
        dp_l = min(dp_preclose, df_dapan.close.min())
        dp_hh = dp_l + (dp_h - dp_l) * 7.5 / 8
        dp_ll = dp_l + (dp_h - dp_l) * 0.5 / 8
        df_dapan.loc[(df_dapan.close < dp_hh) & (df_dapan.close.shift(1) > dp_hh), 'crossdw'] = -0.5
        df_dapan.loc[(df_dapan.close > dp_ll) & (df_dapan.close.shift(1) < dp_ll), 'crossup'] = -0.5
        df_dapan['cp30'] = (df_dapan['close'] - df_dapan['close'].rolling(30).min()) / (
                df_dapan['close'].rolling(30).max() - df_dapan['close'].rolling(30).min())
        df_dapan['cp60'] = (df_dapan['close'] - df_dapan['close'].rolling(60).min()) / (
                df_dapan['close'].rolling(60, min_periods=31).max() - df_dapan['close'].rolling(60, min_periods=31).min())
        df_dapan['cm10'] = df_dapan['close'].rolling(10).mean()
        df_dapan['cm20'] = df_dapan['close'].rolling(20).mean()
        df_dapan['cabovem10'] = df_dapan['close'] > df_dapan['cm10']
        df_dapan.loc[(df_dapan['cp30'] < 0.3) & (df_dapan['cp60'] < 0.3) & (df_dapan['cabovem10'] == True) & \
                    (df_dapan['close'] > df_dapan['close'].shift(1)), 'pivotup'] = 0.5
        df_dapan.loc[(df_dapan['cp30'] > 0.7) & (df_dapan['cp60'] > 0.7) & (df_dapan['cabovem10'] == False) & \
                    (df_dapan['close'] < df_dapan['close'].shift(1)), 'pivotdw'] = 0.5

        df_etf1min = self.get_ETF_data()
        df_etf1min.reset_index(drop=False, inplace=True)
        df_etf1min['datetime'] = df_etf1min['datetime'].apply(lambda x: x.replace('13:00','11:30'))

        df_zdjs = self.tdx_data.get_kline_data('880005', backset=0, klines=300, period=8)
        df_zdjs.rename(columns={'close': 'zdjs'}, inplace=True)
        tmp = df_zdjs[df_zdjs['datetime'].str.contains('15:00')]
        if '15:00' in df_zdjs['datetime'].values[-1]:
            preidx = tmp.index[-2]
        else:
            preidx = tmp.index[-1]
        # preidx = tmp.index[0]
        df_zdjs = df_zdjs[preidx:]
        df_zdjs.reset_index(drop=True, inplace=True)
        self.zdjs = df_zdjs['zdjs'].values[-1]
        df_zdjs['datetime'] = df_zdjs['datetime'].apply(lambda x: x.replace('13:00', '11:30'))
        df_all = pd.merge(df_zdjs[['datetime', 'zdjs']], df_dapan,  on='datetime', how='left')
        df_temp = pd.merge(df_all, df_etf1min, on='datetime', how='left')



        for k, v in self.etf_dict2.items():

            if ('amount',k) in df_temp.columns:
                tmp = df_temp[[('close', k), ('volume', k),('amount',k)]]
                tmp['amtcum'] = tmp[('amount',k)].cumsum()
            else:
                tmp=df_temp[[('close',k),('volume',k)]]
                tmp['amt'] = tmp[('close',k)]*tmp[('volume',k)]
                tmp['amtcum'] = tmp['amt'].cumsum()
            tmp['volcum'] = tmp[('volume', k)].cumsum()
            tmp[f'avg_{k}'] = tmp['amtcum']/tmp['volcum']
            df_temp[f'avg_{k}'] = tmp[f'avg_{k}']
            df_temp[f'amtcum_{k}'] = tmp['amtcum']
            df_temp[f'bosspct_{k}'] = df_temp[('boss', k)]/df_temp[f'amtcum_{k}']*100000000*10
            df_temp[('bossm10',k)] = df_temp[('boss', k)].rolling(10).mean()

            pre_ccb = df_temp[('pre_ccb',k)].values[-1]
            ccb_open = df_temp[('ccb_open',k)].values[-1]
            # df_temp[('ccb_pct',k)] = df_temp[('ccb',k)].apply(lambda x: x / pre_ccb * 1 - 1)
            # df_temp[('ccb_open2pct',k)] = df_temp[('ccb',k)].apply(lambda x: x / ccb_open * 1 - 1)
            # df_temp[('ccb_above_open',k)] = df_temp[('ccb',k)].apply(lambda x: 1 if x > ccb_open else 0)
            df_temp.loc[(df_temp[('cm20up',k)] == 1) & (df_temp[('ccbm20up',k)] == 0) & (df_temp[('ccb_above_open',k)] == 0) & (
                        (df_temp[('ccb_pct',k)] < -0.03) | (df_temp[('ccb_open2pct',k)] < -0.03)), f'c_enterlong_{k}'] = df_temp[('close',k)]

            df_temp.loc[(df_temp[('cm20up',k)] == 0) & (df_temp[('ccbm20up',k)] == 1) & (df_temp[('ccb_above_open',k)] == 1) & (
                        (df_temp[('ccb_pct',k)] > 0.03) | (df_temp[('ccb_open2pct',k)] > 0.03)), f'c_entershort_{k}'] = df_temp[('close',k)]


        ktime = df_temp['datetime'].values[-1][2:].replace('-','').replace(' ','_')
        stamp = datetime.now().strftime('%H:%M:%S')
        timetitle = f'{ktime}--时间戳 {stamp}'

        dp_boss = df_temp['boss'].ffill().values[-1]/100000000
        dp_amount = df_temp['amttrend'].ffill().values[-1]/100000000
        self.dp_boss = dp_boss
        self.dp_amount = dp_amount
        df_temp['dpamtcum'] = df_temp['allamt'].cumsum()
        df_temp['dpbosspct'] = df_temp['boss']/df_temp['dpamtcum']*100
        dp_amtcum = df_temp['dpamtcum'].ffill().values[-1] / 100000000
        dp_bosspct = df_temp['dpbosspct'].ffill().values[-1]
        df_temp['sig'] = df_temp[['pivotup', 'pivotdw', 'crossup', 'crossdw']].notnull().any(axis=1)
        df_temp['sig'] = df_temp.apply(lambda x: np.nan if x.sig == False else x.close, axis=1)
        df_temp['sig'] = df_temp['sig'].ffill()
        df_temp['up'] = df_temp.apply(lambda x: 0 if x.close > x.sig else np.nan, axis=1)
        df_temp['dw'] = df_temp.apply(lambda x: 0 if x.close < x.sig else np.nan, axis=1)
        df_temp['bossm10'] = df_temp['boss'].rolling(10).mean()

        # boss = df_temp['boss'].values[-1]
        # bossr1 = df_temp['boss'].values[-2]
        # bossm10 = df_temp['bossm10'].values[-1]
        # bossm10r1 = df_temp['bossm10'].values[-2]

        # if boss>bossm10 and bossr1<bossm10r1:
        #     playsound('utils\\morning.mp3')
        #     print('主力资金上穿均线')
        #     if len(pushurl)>10:
        #         msgURL = pushurl + '主力资金上穿均线'
        #         requests.get(msgURL)
        # elif boss<bossm10 and bossr1>bossm10r1:
        #     playsound('utils\\swoosh.mp3')
        #     print('主力资金下穿均线')
        #     if len(pushurl) > 10:
        #         msgURL = pushurl + '主力资金下穿均线'
        #         requests.get(msgURL)
        # else:
        #     pass

        self.dp_amtcum = df_temp['dpamtcum'].ffill().values[-1] / 100000000
        self.dp_bosspct = df_temp['dpbosspct'].ffill().values[-1]
        self.data = df_temp

        # return df_temp