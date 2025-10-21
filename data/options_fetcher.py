# data/options_fetcher.py
import pandas as pd
import numpy as np
import re,os
import json
import time
import requests
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed




class OptionsDataFetcher:
    def __init__(self, tdx_data,etf_dict,etf_dict2,opt_path):
        self.tdx_data = tdx_data
        self.etf_dict = etf_dict
        self.etf_dict2 = etf_dict2
        self.opt_path = opt_path

    @retry(stop=stop_after_attempt(1), wait=wait_fixed(5))
    def safe_get_request(self,url, headers=None, params=None):
        return requests.get(url, headers=headers, params=params, proxies={})


    def get_all_options_v3(self,cookie=None) -> pd.DataFrame:
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            "Accept": "gzip, deflate, br, zstd",
            "Connection": "keep-alive",
            "Cookie": cookie if cookie is not None else "qgqp_b_id=435b18200eebe2cbb5bdd3b3af2db1b1; intellpositionL=522px; intellpositionT=1399.22px; em_hq_fls=js; pgv_pvi=6852699136; st_pvi=73734391542044; st_sp=2020-07-27%2010%3A10%3A43; st_inirUrl=http%3A%2F%2Fdata.eastmoney.com%2Fhsgt%2Findex.html",
            "Host": "push2.eastmoney.com",
            "Referer": "https://data.eastmoney.com/other/valueAnal.html",
        }

        field_map0 = {'f12': 'code', 'f14': 'name', 'f301': '到期日', 'f331': 'ETFcode', 'f333': 'ETFname',
                      'f2': 'close', 'f3': 'pctChg', 'f334': 'ETFprice', 'f335': 'ETFpct', 'f337': '平衡价',
                      'f250': '溢价率', 'f161': '行权价'}  # ,'f47':'vol','f48':'amount','f133':'allvol'}

        params = {
            'cb': f'jQuery112305886670098015102_{str(int(time.time() * 1000))}',
            'fid': 'f301',
            'po': '1',
            'pz': '100',
            'pn': '1',
            'np': '1',
            'fltt': '2',
            'invt': '2',
            'ut': '8dec03ba335b81bf4ebdf7b29ec27d15',
            'fields': 'f1,f2,f3,f12,f13,f14,f161,f250,f330,f331,f332,f333,f334,f335,f337,f301,f152',
            'fs': 'm:10'
        }

        #header['Cookie'] = "qgqp_b_id=1296411ee76ff7730dfcad6866f999b1; st_nvi=9tPzGRrddxABSsl2PgP1n272a; st_si=23335772256745; nid=0fb1d9c49b03318096b53bba0088ecd4; nid_create_time=1760509594771; gvi=waQTWnkGHmybhqeOUOulY2d86; gvi_create_time=1760509594771; fullscreengg=1; fullscreengg2=1; wsc_checkuser_ok=1; st_pvi=99659171718134; st_sp=2020-12-25%2019%3A43%3A29; st_inirUrl=http%3A%2F%2Ffinance.eastmoney.com%2Fa%2F202002261396627810.html; st_sn=12; st_psi="

        url1 = 'https://push2.eastmoney.com/api/qt/clist/get'
        # url1 = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112307429657982724098_1687701611430&fid=f250' + \
        #        '&po=1&pz=200&pn=1&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5' + \
        #        '&fields=f1,f2,f3,f12,f13,f14,f161,f250,f330,f331,f332,f333,f334,f335,f337,f301,f152&fs=m:10'
        #   requests.get(url1, headers=header, params=params, proxies={})

        try:
            res = self.safe_get_request(url1, headers=header, params=params)
            tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"', '"0"')

            data1 = pd.DataFrame(json.loads(tmp)['data']['diff'])
            data1.rename(columns=field_map0, inplace=True)

            for i in range(2, 10, 1):
                # url1i = url1.replace('&pn=1&', f'&pn={i}&')
                # resi = self.safe_get_request(url1i, headers=header)
                params['pn'] = str(i)
                resi = self.safe_get_request(url1, headers=header, params=params)
                if len(resi.text) > 500:
                    tmpi = re.search(r'^\w+\((.*)\);$', resi.text).group(1).replace('"-"', '"0"')
                    data1i = pd.DataFrame(json.loads(tmpi)['data']['diff'])
                    data1i.rename(columns=field_map0, inplace=True)
                    if len(data1i) ==100:
                        data1 = pd.concat([data1, data1i])
                    elif len(data1i)>0:
                        data1 = pd.concat([data1, data1i])
                        break
                    else:
                        break
        except Exception as e:
            print(f"get_all_options_v3 data1 error: {e}")
            data1 = pd.DataFrame()
            return pd.DataFrame()

        try:
            # url2 = url1[:-1] + '2'
            params['fs'] = 'm:12'
            params['pn'] = '1'
            # #   requests.get(url1, headers=header, params=params, proxies={})
            res = self.safe_get_request(url1, headers=header, params=params)  # , headers=header)

            tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"', '"0"')
            data2 = pd.DataFrame(json.loads(tmp)['data']['diff'])
            data2.rename(columns=field_map0, inplace=True)

            for i in range(2,10, 1):
                # url1i = url2.replace('&pn=1&', f'&pn={i}&')
                # resi = self.safe_get_request(url1i, headers=header)
                params['pn'] = str(i)
                resi = self.safe_get_request(url1, headers=header, params=params)
                if len(resi.text) > 500:
                    tmpi = re.search(r'^\w+\((.*)\);$', resi.text).group(1).replace('"-"', '"0"')
                    data2i = pd.DataFrame(json.loads(tmpi)['data']['diff'])
                    data2i.rename(columns=field_map0, inplace=True)
                    if len(data2i) == 100:
                        data2 = pd.concat([data2, data2i])
                    elif len(data2i) > 0:
                        data2 = pd.concat([data2, data2i])
                        break
                    else:
                        break
        except Exception as e:
            print(f"get_all_options_v3 data2 error: {e}")
            data2 = pd.DataFrame()
            return pd.DataFrame()

        if len(data1) == 0 or len(data2) == 0:
            return pd.DataFrame()

        data = pd.concat([data1, data2])
        data = data[list(field_map0.values())]

        data['market'] = data['ETFcode'].apply(lambda x: '沪市' if x[0] == '5' else '深市')
        data['direction'] = data['name'].apply(lambda x: 'call' if '购' in x else 'put')
        data['due_date'] = data['到期日'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').date())
        data['dte'] = data['due_date'].apply(lambda x: (x - datetime.now().date()).days)
        data['close'] = data['close'].astype(float)
        data['到期日'] = data['到期日'].astype(str)
        data['行权pct'] = data.apply(lambda x: round(x['行权价'] / x['ETFprice'] * 100 - 100, 2), axis=1)
        data['itm'] = data.apply(
            lambda x: max(0, x.ETFprice - x['行权价']) if x.direction == 'call' else max(0, x['行权价'] - x.ETFprice),
            axis=1)
        data['otm'] = data.apply(lambda x: x.close - x.itm, axis=1)

        return data

    def get_my_options(self, filter_dict,cookie=None):
        # if self.logger is not None:
        #     self.logger.info(f'executing get_my_options...')

        now = self.tdx_data.get_kline_data('999999', 0, 20, 8)

        current_datetime = datetime.strptime(now['datetime'].values[-1], '%Y-%m-%d %H:%M')

        earlymorning = True if (time.strftime("%H%M", time.localtime()) >= '0900' and time.strftime("%H%M",
                                time.localtime()) <= '0935') else False
        tempdates = self.tdx_data.get_kline_data('399001', 0, 20, 9)
        opt_fn_last = self.opt_path + '\\沪深期权清单_' + tempdates['datetime'].values[-2][:10].replace('-', '') + '.csv'
        opt_fn = self.opt_path + '\\沪深期权清单_' + now['datetime'].values[-1][:10].replace('-', '') + '.csv'

        if earlymorning and os.path.exists(opt_fn_last):
            data = pd.read_csv(opt_fn_last, encoding='gbk', dtype={'ETFcode': str, 'code': str})
        elif os.path.exists(opt_fn):
            modified_timestamp = os.path.getmtime(opt_fn)
            modified_datetime = datetime.fromtimestamp(modified_timestamp)
            time_delta = current_datetime - modified_datetime
            gap_seconds = time_delta.days * 24 * 3600 + time_delta.seconds
            if gap_seconds < float(filter_dict['opt_screen_frequency']):  #1800:
                print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} reusing option file')
                data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
            else:
                try:
                    data = self.get_all_options_v3(cookie=cookie)
                    print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} Try New option file')
                    if len(data)<=900:
                        print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} new data rows {len(data)}, reusing option file')
                        data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
                    else:
                        data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
                except:
                    print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} update failed, reusing option file')
                    data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
        else:
            print('New option file ' + opt_fn)
            data = self.get_all_options_v3(cookie=cookie)
            if len(data)>900:
                data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
            else:
                data = pd.read_csv(opt_fn_last, encoding='gbk', dtype={'ETFcode': str, 'code': str})
                print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} new data<900, using yesterday file')

        return self.tdx_data.getTradeOptions(list(self.etf_dict.keys()), data, closeMean=float(filter_dict['close_mean'])), data


    def get_option_data(self,png_dict, df_optlist,df_etf):

        period = 8
        klines = 300

        df_options = pd.DataFrame()
        new_optlist = {}

        for k, v in self.etf_dict2.items():
            optLongCode = png_dict[k].split('\n')[0].split(':')[1].split('_')[0]
            optShortCode = png_dict[k].split('\n')[1].split(':')[1].split('_')[0]

            ETFprice = self.tdx_data.get_kline_data(v, backset=0, klines=klines, period=period)['close'].values[-1]
            longrow = df_optlist.loc[df_optlist['code'] == optLongCode]
            shortrow = df_optlist.loc[df_optlist['code'] == optShortCode]

            df_long = self.tdx_data.get_kline_data(optLongCode, backset=0, klines=klines, period=period)
            # print(k,v,optLongCode, 'df_long len:', len(df_long))

            df_long['cp30'] = (df_long['close'] - df_long['close'].rolling(30).min()) / (
                    df_long['close'].rolling(30).max() - df_long['close'].rolling(30).min())
            df_long['cp60'] = (df_long['close'] - df_long['close'].rolling(60).min()) / (
                    df_long['close'].rolling(60, min_periods=31).max() - df_long['close'].rolling(60, min_periods=31).min())
            df_long['cm10'] = df_long['close'].rolling(10).mean()
            df_long['cm20'] = df_long['close'].rolling(20).mean()
            df_long['cabovem10'] = df_long['close'] > df_long['cm10']

            df_long['longm20'] = df_long['close'].rolling(20).mean()

            if len(df_long) <= 240:
                preclose_long = df_long['close'].values[0]
            else:
                tmp = df_long[df_long['datetime'].str.contains('15:00')]
                if '15:00' in df_long['datetime'].values[-1]:
                    preidx = tmp.index[-2]
                else:
                    preidx = tmp.index[-1]
                df_long = df_long.iloc[preidx:]
                preclose_long = df_long['close'].values[0]

            k_h = max(preclose_long, df_long['high'].max())
            k_l = min(preclose_long, df_long['low'].min())
            k_hh = k_l + (k_h - k_l) * 7.5 / 8
            k_ll = k_l + (k_h - k_l) * 0.5 / 8
            df_long['Long_crossdw'] = np.nan
            df_long['Long_crossup'] = np.nan
            df_long.loc[(df_long['close'] < k_hh) & (df_long['close'].shift(1) > k_hh), 'Long_crossdw'] = preclose_long
            df_long.loc[(df_long['close'] > k_ll) & (df_long['close'].shift(1) < k_ll), 'Long_crossup'] = preclose_long

            df_long.loc[(df_long['cp30'] < 0.3) & (df_long['cp60'] < 0.3) & (df_long['cabovem10'] == True) & \
                        (df_long['close'] > df_long['close'].shift(1)), 'Long_pivotup'] = preclose_long
            df_long.loc[(df_long['cp30'] > 0.7) & (df_long['cp60'] > 0.7) & (df_long['cabovem10'] == False) & \
                        (df_long['close'] < df_long['close'].shift(1)), 'Long_pivotdw'] = preclose_long

            df_short = self.tdx_data.get_kline_data(optShortCode, backset=0, klines=klines, period=period)
            # print(k, v, optShortCode, 'df_short len:', len(df_short))
            df_short['cp30'] = (df_short['close'] - df_short['close'].rolling(30).min()) / (
                    df_short['close'].rolling(30).max() - df_short['close'].rolling(30).min())
            df_short['cp60'] = (df_short['close'] - df_short['close'].rolling(60).min()) / (
                    df_short['close'].rolling(60, min_periods=31).max() - df_short['close'].rolling(60,
                                                                                                    min_periods=31).min())
            df_short['cm10'] = df_short['close'].rolling(10).mean()
            df_short['cm20'] = df_short['close'].rolling(20).mean()
            df_short['cabovem10'] = df_short['close'] > df_short['cm10']

            df_short['shortm20'] = df_short['close'].rolling(20).mean()
            if len(df_short) <= 240:
                preclose_short = df_short['close'].values[0]
            else:
                tmp = df_short[df_short['datetime'].str.contains('15:00')]
                if '15:00' in df_short['datetime'].values[-1]:
                    preidx = tmp.index[-2]
                else:
                    preidx = tmp.index[-1]
                df_short = df_short.iloc[preidx:]

                preclose_short = df_short['close'].values[0]
            k_h = max(preclose_short, df_short['high'].max())
            k_l = min(preclose_short, df_short['low'].min())
            k_hh = k_l + (k_h - k_l) * 7.5 / 8
            k_ll = k_l + (k_h - k_l) * 0.5 / 8
            df_short['Short_crossdw'] = np.nan
            df_short['Short_crossup'] = np.nan
            df_short.loc[(df_short['close'] < k_hh) & (df_short['close'].shift(1) > k_hh), 'Short_crossdw'] = preclose_short
            df_short.loc[(df_short['close'] > k_ll) & (df_short['close'].shift(1) < k_ll), 'Short_crossup'] = preclose_short

            df_short.loc[(df_short['cp30'] < 0.3) & (df_short['cp60'] < 0.3) & (df_short['cabovem10'] == True) & \
                         (df_short['close'] > df_short['close'].shift(1)), 'Short_pivotup'] = preclose_short
            df_short.loc[(df_short['cp30'] > 0.7) & (df_short['cp60'] > 0.7) & (df_short['cabovem10'] == False) & \
                         (df_short['close'] < df_short['close'].shift(1)), 'Short_pivotdw'] = preclose_short

            df_long.rename(columns={'close': 'long'}, inplace=True)
            df_short.rename(columns={'close': 'short'}, inplace=True)
            df_opt = pd.merge(
                df_long[['datetime', 'long', 'longm20', 'Long_crossdw', 'Long_crossup', 'Long_pivotup', 'Long_pivotdw']],
                df_short[
                    ['datetime', 'short', 'shortm20', 'Short_crossdw', 'Short_crossup', 'Short_pivotup', 'Short_pivotdw']],
                on='datetime', how='inner')
            ttt = df_etf[['datetime', ('up', k), ('dw', k), ('boss', k), ('bossm10', k), f'c_enterlong_{k}', f'c_entershort_{k}']]
            ttt.rename(columns={('up', k): 'up', ('dw', k): 'dw', ('boss', k): 'boss', ('bossm10', k): 'bossm10',
                                f'c_enterlong_{k}': 'c_enterlong', f'c_entershort_{k}': 'c_entershort'},
                       inplace=True)
            df_opt = pd.merge(df_opt, ttt, on='datetime', how='left')
            df_opt['enterlong'] = df_opt.apply(lambda x: x.long if ~np.isnan(x['c_enterlong']) else np.nan, axis=1)
            df_opt['entershort'] = df_opt.apply(lambda x: x.long if ~np.isnan(x['c_entershort']) else np.nan, axis=1)
            df_opt['up'] = df_opt['up'].replace(0.0, preclose_long)
            df_opt['dw'] = df_opt['dw'].replace(0.0, preclose_long)

            df_opt['etf'] = k
            df_options = pd.concat([df_options, df_opt])

            long_itm = max(0, ETFprice - longrow['行权价'].values[0])
            long_otm = df_opt['long'].values[-1] - long_itm
            short_itm = max(0, shortrow['行权价'].values[0] - ETFprice)
            short_otm = df_opt['short'].values[-1] - short_itm

            longtext = f'''认购:{optLongCode}_{longrow['name'].values[0]}_{df_opt['long'].values[-1]:.4f}=itm{long_itm * 10000:.0f}+{long_otm * 10000:.0f}''' #_金额:{(df_long['long'] * df_long['trade']).sum():.0f}万'''
            shorttext = f'''认沽:{optShortCode}_{shortrow['name'].values[0]}_{df_opt['short'].values[-1]:.4f}=itm{short_itm * 10000:.0f}+{short_otm * 10000:.0f}'''#_金额:{(df_short['short'] * df_short['trade']).sum():.0f}万'''
            new_optlist[k] = f'''{longtext}\n{shorttext}'''

        opt_Pivot = df_options.pivot_table(index='datetime', columns='etf', values=['long', 'longm20', 'short', 'shortm20',
                    'Long_crossdw', 'Long_crossup', 'Short_crossdw', 'Short_crossup',
                    'Long_pivotup', 'Long_pivotdw','Short_pivotup', 'Short_pivotdw', 'up',
                    'dw', 'boss', 'bossm10','enterlong', 'entershort'], dropna=False)
        self.new_optlist = new_optlist
        self.data = opt_Pivot