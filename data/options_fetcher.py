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

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(5))
    def safe_get_request(self,url, headers=None, params=None):
        return requests.get(url, headers=headers, params=params, proxies={})
    def get_options_tformat(self, df_4T: pd.DataFrame) -> pd.DataFrame:
        field_map3 = {
            'f14': 'Cname', 'f12': 'Ccode', 'f2': 'Cprice', 'f3': 'CpctChg',
            'f4': 'C涨跌额', 'f108': 'C持仓量', 'f5': 'Cvol', 'f249': 'Civ', 'f250': 'C折溢价率', 'f161': '行权价',
            'f340': 'Pname', 'f339': 'Pcode', 'f341': 'Pprice', 'f343': 'PpctChg', 'f342': 'P涨跌额',
            'f345': 'P持仓量', 'f344': 'Pvol', 'f346': 'Piv', 'f347': 'P折溢价率'
        }

        df_T_data = pd.DataFrame()
        try:
            for etfcode, expiredate in zip(df_4T['ETFcode'], df_4T['到期日']):
                code = '1.' + etfcode if etfcode[0] == '5' else '0.' + etfcode
                url3 = f'https://push2.eastmoney.com/api/qt/slist/get?cb=jQuery112400098284603835751_1695513185234&secid={code}&exti={expiredate[:6]}&spt=9&fltt=2&invt=2&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fields=f1,f2,f3,f4,f5,f12,f13,f14,f108,f152,f161,f249,f250,f330,f334,f339,f340,f341,f342,f343,f344,f345,f346,f347&fid=f161&pn=1&pz=20&po=0&wbp2u=|0|0|0|web&_=1695513185258'
                res = self.safe_get_request(url3)
                tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
                single = pd.DataFrame(json.loads(tmp)['data']['diff'])
                df_T_data = pd.concat([df_T_data, single], ignore_index=True)

            df_T_data.rename(columns=field_map3, inplace=True)
            return df_T_data[list(field_map3.values())]

        except Exception as e:
            print(f"Error in get_options_tformat: {e}")
            return pd.DataFrame()


    def get_options_risk_data(self) -> pd.DataFrame:
        field_map4 = {"f2": "最新价", "f3": "涨跌幅", "f12": "code", "f14": "name", "f301": "到期日",
                      "f302": "杠杆比率", "f303": "实际杠杆", "f325": "Delta", "f326": "Gamma", "f327": "Vega",
                      "f328": "Theta", "f329": "Rho"}

        df_risk = pd.DataFrame()
        try:
            for i in range(1, 11, 1):
                url4 = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112308418460865815227_1695516975860&fid=f3&po=1&' + \
                       'pz=' + '50' + '&pn=' + str(i) + '&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5' + \
                       '&fields=f1,f2,f3,f12,f13,f14,f302,f303,f325,f326,f327,f329,f328,f301,f152,f154&fs=m:10'
                res = self.safe_get_request(url4)
                tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"', '"0"')
                if len(tmp) < 100:
                    continue
                single = pd.DataFrame(json.loads(tmp)['data']['diff'])
                df_risk = pd.concat([df_risk, single])

            df_risk.rename(columns=field_map4, inplace=True)
            # df_risk = df_risk[list(field_map4.values())]

            return df_risk[list(field_map4.values())]
        except Exception as e:
            print(f"Error in get_options_risk_data: {e}")
            return pd.DataFrame()

    # def get_all_options_v3bak(self,cookie=None) -> pd.DataFrame:
    #     header = {
    #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0',
    #         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    #         "Cookie": cookie if cookie is not None else "qgqp_b_id=2e5f72b5a8e430f5f578199b44b2c650; st_nvi=QTo1DhRK-9b0WLzl763GF90fb; nid=005c4e7016dc8547c46bc58c842215af; nid_create_time=1754291547115; gvi=KrcKF3koRrZAdqQD855wDc6b3; gvi_create_time=1754291547115; mtp=1; ct=cI7Lo5aNpHXvsN7fQyYXistdSwQypZ6pCRwLMoxfH0iERvj4LVcqu917dRLv7P67zYRdhnod-iTNFzaJa_kSIEtIDFUZn_1C8_NHoD_Y5RKRBCYJYOZwT5neykaLga0PXwdy35PQZGq_4CFY3lRy-_qbkh7idXA7TLSHoOJ-TPo; ut=FobyicMgeV63FIqRjcfDeRRI5bofMxGbDM6MboJ-5VcfKNE1KSeQpn0nTjA4MCb02RzK21f9bRjUYAzm70AllVVVIdtMEotGI_s9g10b0UaKP69Thj2H5QcD1ZiZSO-OEilTSeDyI7ovhg9WIMkOWNAaNzej3GSP1SsySJd2uveX0gCto16y4-lYWnQCYc7ToqxcAYY3c-nDrc7cseHQ_o8fLJc2yUu_KeKuEZfiOMyUYm0sP5ZFQPNbs-8sWRqxoZq2gmtlQV0x3BUHmu1HgVrSbeyYDla2; pi=2186316006663558%3Bu2186316006663558%3B%E8%82%A1%E5%8F%8BDOk6GW%3BYNLvxdi0J24eaLlSY08HJwyb5HP5SUwF2j7%2FqoTM2s6oi6XGG3xHh3NoXD44aZOh%2Fq%2BdgQuAKtcnwn95MBNB7pvGwFHdKbbmxLeeKSK6KPN%2Bg%2FzZz6CFR71Kdfy45FQE3JCvNVotzQnXQbfXHrjG4wMjYpLYsPktYSGL6ycrDQdESRNzfOEgfq3iunRh6Y8ZHfoh%2BXCG%3BgzA1ItCCoCphbnEKpXwGOtx2wcjvzwYLQ0gL6%2B9lwUMiY5IPkp%2BloqrpW6WGzsIscBnu8lrcdhaidpNjazeWLFjyNXpQwjzkheO9Ds8%2FuI3lwnEiMf3ay%2BffRVKbX0OmFM5iiXwsTn6iEPTV3ik2mK29yhMoeg%3D%3D; uidal=2186316006663558%e8%82%a1%e5%8f%8bDOk6GW; sid=152379750; vtpst=|; fullscreengg=1; fullscreengg2=1; st_si=15442846996330; st_asi=delete; st_pvi=99659171718134; st_sp=2020-12-25%2019%3A43%3A29; st_inirUrl=http%3A%2F%2Ffinance.eastmoney.com%2Fa%2F202002261396627810.html; st_sn=12; st_psi=20251014165507189-113300300871-4181137287",
    #         "Host": "push2.eastmoney.com",
    #     }
    #
    #     field_map0 = {'f12': 'code', 'f14': 'name', 'f301': '到期日', 'f331': 'ETFcode', 'f333': 'ETFname',
    #                   'f2': 'close', 'f3': 'pctChg', 'f334': 'ETFprice', 'f335': 'ETFpct', 'f337': '平衡价',
    #                   'f250': '溢价率', 'f161': '行权价'}  # ,'f47':'vol','f48':'amount','f133':'allvol'}
    #
    #     url1 = 'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112307429657982724098_1687701611430&fid=f250' + \
    #            '&po=1&pz=200&pn=1&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5' + \
    #            '&fields=f1,f2,f3,f12,f13,f14,f161,f250,f330,f331,f332,f333,f334,f335,f337,f301,f152&fs=m:10'
    #
    #     try:
    #         res = self.safe_get_request(url1, headers=header)
    #         tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"', '"0"')
    #
    #         data1 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    #         data1.rename(columns=field_map0, inplace=True)
    #
    #         for i in range(2, 6, 1):
    #             url1i = url1.replace('&pn=1&', f'&pn={i}&')
    #             resi = self.safe_get_request(url1i, headers=header)
    #             if len(resi.text) > 500:
    #                 tmpi = re.search(r'^\w+\((.*)\);$', resi.text).group(1).replace('"-"', '"0"')
    #                 data1i = pd.DataFrame(json.loads(tmpi)['data']['diff'])
    #                 data1i.rename(columns=field_map0, inplace=True)
    #                 if len(data1i) > 0:
    #                     data1 = pd.concat([data1, data1i])
    #     except Exception as e:
    #         print(f"get_all_options_v3 data1 error: {e}")
    #
    #     url2 = url1[:-1] + '2'
    #     res = self.safe_get_request(url2, headers=header)#, headers=header)
    #
    #     try:
    #         tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"', '"0"')
    #         data2 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    #         data2.rename(columns=field_map0, inplace=True)
    #
    #         for i in range(2, 6, 1):
    #             url1i = url2.replace('&pn=1&', f'&pn={i}&')
    #             resi = self.safe_get_request(url1i, headers=header)
    #             if len(resi.text) > 500:
    #                 tmpi = re.search(r'^\w+\((.*)\);$', resi.text).group(1).replace('"-"', '"0"')
    #                 data2i = pd.DataFrame(json.loads(tmpi)['data']['diff'])
    #                 data2i.rename(columns=field_map0, inplace=True)
    #                 if len(data2i) > 0:
    #                     data2 = pd.concat([data2, data2i])
    #     except Exception as e:
    #         print(f"get_all_options_v3 data2 error: {e}")
    #
    #     data = pd.concat([data1, data2])
    #     data = data[list(field_map0.values())]
    #
    #     data['market'] = data['ETFcode'].apply(lambda x: '沪市' if x[0] == '5' else '深市')
    #     data['direction'] = data['name'].apply(lambda x: 'call' if '购' in x else 'put')
    #     data['due_date'] = data['到期日'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').date())
    #     data['dte'] = data['due_date'].apply(lambda x: (x - datetime.now().date()).days)
    #     data['close'] = data['close'].astype(float)
    #     data['到期日'] = data['到期日'].astype(str)
    #     data['行权pct'] = data.apply(lambda x: round(x['行权价'] / x['ETFprice'] * 100 - 100, 2), axis=1)
    #     data['itm'] = data.apply(
    #         lambda x: max(0, x.ETFprice - x['行权价']) if x.direction == 'call' else max(0, x['行权价'] - x.ETFprice),
    #         axis=1)
    #     data['otm'] = data.apply(lambda x: x.close - x.itm, axis=1)
    #
    #     df_4T = data.pivot_table(index=['ETFcode', '到期日'], values=['name'], aggfunc=['count']).reset_index()
    #     df_4T.columns = ['ETFcode', '到期日', '数量']
    #
    #     df_T_format = self.get_options_tformat(df_4T)
    #     if len(df_T_format) >0:
    #         tempc = df_T_format[['Ccode', 'C持仓量', 'Cvol']]
    #         tempc.columns = ['code', '持仓量', 'vol']
    #         tempp = df_T_format[['Pcode', 'P持仓量', 'Pvol']]
    #         tempp.columns = ['code', '持仓量', 'vol']
    #         temp = pd.concat([tempc, tempp])
    #         temp['vol'] = temp['vol'].astype(int)
    #         data = pd.merge(data, temp, on='code', how='left')
    #         data['amount'] = data['close'] * data['vol']
    #
    #     df_risk = self.get_options_risk_data()
    #     if len(df_risk) > 0:
    #         data = pd.merge(data, df_risk[['code', '杠杆比率', '实际杠杆', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']],
    #                         on='code', how='left')
    #     return data


    # def get_my_options(self, filter_dict, cookie=None):
    #
    #     # now = pd.DataFrame(api.get_index_bars(8, 1, '999999', 0, 20))
    #     now = self.tdx_data.get_kline_data('999999', 0, 20, 8)
    #
    #     current_datetime = datetime.strptime(now['datetime'].values[-1], '%Y-%m-%d %H:%M')
    #
    #     earlymorning = True if (time.strftime("%H%M", time.localtime()) >= '0930' and time.strftime("%H%M",
    #                             time.localtime()) <= '0935') else False
    #     tempdates = self.tdx_data.get_kline_data('399001', 0, 20, 9)
    #     opt_fn_last = self.opt_path + '\\沪深期权清单_' + tempdates['datetime'].values[-2][:10].replace('-', '') + '.csv'
    #     opt_fn = self.opt_path + '\\沪深期权清单_' + now['datetime'].values[-1][:10].replace('-', '') + '.csv'
    #
    #     if earlymorning and os.path.exists(opt_fn_last):
    #         data = pd.read_csv(opt_fn_last, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    #     elif os.path.exists(opt_fn):
    #         modified_timestamp = os.path.getmtime(opt_fn)
    #         modified_datetime = datetime.fromtimestamp(modified_timestamp)
    #         time_delta = current_datetime - modified_datetime
    #         gap_seconds = time_delta.days * 24 * 3600 + time_delta.seconds
    #         if gap_seconds < 1000:
    #             print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} reusing option file')
    #             data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    #         else:
    #             try:
    #                 data = self.get_all_options_v3(cookie=cookie)
    #                 print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} Try New option file')
    #                 if len(data) <= 900:
    #                     print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} new data<900, reusing option file')
    #                     data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    #                 else:
    #                     data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
    #             except:
    #                 print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} update failed, reusing option file')
    #                 data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    #     else:
    #         print('New option file ' + opt_fn)
    #         data = self.get_all_options_v3(cookie=cookie)
    #         if len(data) > 900:
    #             data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
    #         else:
    #             data = pd.read_csv(opt_fn_last, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    #             print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} new data<900, using yesterday file')
    #
    #     data.fillna(0, inplace=True)
    #     # amtlist = data['amount'].values.tolist()
    #     # amtlist.sort(ascending=False)
    #     # amtthreshold = amtlist[200] if len(amtlist)>200 else amtlist[-1]
    #     # data.sort_values(by='amount', ascending=False, inplace=True)
    #     data['itm'] = data.apply(
    #         lambda x: max(0, x.ETFprice - x['行权价']) if x.direction == 'call' else max(0, x['行权价'] - x.ETFprice),
    #         axis=1)
    #     data['otm'] = data.apply(lambda x: x.close - x.itm, axis=1)
    #
    #     # data2 = data[data['amount']>amtthreshold]
    #
    #     png_dict = {}
    #     for key,etfcode in self.etf_dict2.items():
    #         # etfcode = etf_dict[key]
    #         tmpdf = data[(data['ETFcode'] == etfcode) & (~data['name'].str.contains('A')) & (
    #                 data['dte'] > int(filter_dict['dte_low'])) & (data['dte'] < int(filter_dict['dte_high']))]
    #         tmpdf['tmpfact'] = tmpdf['close'].apply(
    #             lambda x: x / float(filter_dict['close_mean']) if x <= float(filter_dict['close_mean']) else float(filter_dict['close_mean']) / x)
    #         tmpdf['tmpfact2'] = tmpdf['tmpfact'] * tmpdf['tmpfact']  # *tmpdf['tmpfact']*tmpdf['amount']
    #         tmpdf.sort_values(by='tmpfact2', ascending=False, inplace=True)
    #         call = tmpdf[(tmpdf['direction'] == 'call')][:1]
    #         put = tmpdf[(tmpdf['direction'] == 'put')][:1]
    #         if len(call) == 0:
    #             tmpstr = f'{key}认购:流动性过滤为空   '
    #         else:
    #             # tmpstr = '认购:' + call['code'].values[0] + '_' + call['name'].values[0] + '_' + str(
    #             #     call['close'].values[0]) + '=itm' + str(int(call['itm'].values[0] * 10000)) + '+' + str(
    #             #     int(call['otm'].values[0] * 10000)) + \
    #             #          ' 金额:' + str(int(call['amount'].values[0]))
    #             tmpstr = f"认购:{call['code'].values[0]}_{call['name'].values[0]}_{call['close'].values[0]:.4f}=itm{int(call['itm'].values[0] * 10000)}+{int(call['otm'].values[0] * 10000)}"# 金额:{int(call['amount'].values[0])}"
    #
    #         if len(put) == 0:
    #             tmpstr += f'\n{key}认沽:流动性过滤为空'
    #         else:
    #             # tmpstr += '\n认沽:' + put['code'].values[0] + '_' + put['name'].values[0] + '_' + str(
    #             #     put['close'].values[0]) + '=itm' + str(int(put['itm'].values[0] * 10000)) + '+' + str(
    #             #     int(put['otm'].values[0] * 10000)) + \
    #             #           ' 金额:' + str(int(put['amount'].values[0]))
    #             tmpstr += f"\n认沽:{put['code'].values[0]}_{put['name'].values[0]}_{put['close'].values[0]:.4f}=itm{int(put['itm'].values[0] * 10000)}+{int(put['otm'].values[0] * 10000)}"# 金额:{int(put['amount'].values[0])}"
    #
    #
    #         png_dict[key] = tmpstr
    #
    #     return png_dict, data

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

        # df_4T = data.pivot_table(index=['ETFcode', '到期日'], values=['name'], aggfunc=['count']).reset_index()
        # df_4T.columns = ['ETFcode', '到期日', '数量']
        #
        # df_T_format = self.get_options_tformat(df_4T)
        # if len(df_T_format) >0:
        #     tempc = df_T_format[['Ccode', 'C持仓量', 'Cvol']]
        #     tempc.columns = ['code', '持仓量', 'vol']
        #     tempp = df_T_format[['Pcode', 'P持仓量', 'Pvol']]
        #     tempp.columns = ['code', '持仓量', 'vol']
        #     temp = pd.concat([tempc, tempp])
        #     temp['vol'] = temp['vol'].astype(int)
        #     data = pd.merge(data, temp, on='code', how='left')
        #     data['amount'] = data['close'] * data['vol']
        #
        # df_risk = self.get_options_risk_data()
        # if len(df_risk) > 0:
        #     data = pd.merge(data, df_risk[['code', '杠杆比率', '实际杠杆', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']],
        #                     on='code', how='left')
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
            if gap_seconds < 1800:
                print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} reusing option file')
                data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
            else:
                try:
                    data = self.get_all_options_v3(cookie=cookie)
                    print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} Try New option file')
                    if len(data)<=900:
                        print(f'{datetime.now().strftime("%m%d_%H:%M:%S")} new data<900, reusing option file')
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

        data.fillna(0, inplace=True)
        # amtlist = data['amount'].values.tolist()
        # amtlist.sort(ascending=False)
        # amtthreshold = amtlist[200] if len(amtlist)>200 else amtlist[-1]
        # data.sort_values(by='amount', ascending=False, inplace=True)
        data['itm'] = data.apply(
            lambda x: max(0, x.ETFprice - x['行权价']) if x.direction == 'call' else max(0, x['行权价'] - x.ETFprice),
            axis=1)
        data['otm'] = data.apply(lambda x: x.close - x.itm, axis=1)

        # data2 = data[data['amount']>amtthreshold]

        png_dict = {}
        for key,etfcode in self.etf_dict2.items():
            # etfcode = etf_dict[key]
            tmpdf = data[(data['ETFcode'] == etfcode) & (~data['name'].str.contains('A')) & (
                    data['dte'] > int(filter_dict['dte_low'])) & (data['dte'] < int(filter_dict['dte_high']))]
            tmpdf['tmpfact'] = tmpdf['close'].apply(
                lambda x: x / float(filter_dict['close_mean']) if x <= float(filter_dict['close_mean']) else float(filter_dict['close_mean']) / x)
            tmpdf['tmpfact2'] = tmpdf['tmpfact'] * tmpdf['tmpfact']  # *tmpdf['tmpfact']*tmpdf['amount']
            tmpdf.sort_values(by='tmpfact2', ascending=False, inplace=True)
            call = tmpdf[(tmpdf['direction'] == 'call')][:1]
            put = tmpdf[(tmpdf['direction'] == 'put')][:1]
            if len(call) == 0:
                tmpstr = f'{key}认购:流动性过滤为空   '
            else:
                # tmpstr = '认购:' + call['code'].values[0] + '_' + call['name'].values[0] + '_' + str(
                #     call['close'].values[0]) + '=itm' + str(int(call['itm'].values[0] * 10000)) + '+' + str(
                #     int(call['otm'].values[0] * 10000)) + \
                #          ' 金额:' + str(int(call['amount'].values[0]))
                tmpstr = f"认购:{call['code'].values[0]}_{call['name'].values[0]}_{call['close'].values[0]:.4f}=itm{int(call['itm'].values[0] * 10000)}+{int(call['otm'].values[0] * 10000)}"# 金额:{int(call['amount'].values[0])}"

            if len(put) == 0:
                tmpstr += f'\n{key}认沽:流动性过滤为空'
            else:
                # tmpstr += '\n认沽:' + put['code'].values[0] + '_' + put['name'].values[0] + '_' + str(
                #     put['close'].values[0]) + '=itm' + str(int(put['itm'].values[0] * 10000)) + '+' + str(
                #     int(put['otm'].values[0] * 10000)) + \
                #           ' 金额:' + str(int(put['amount'].values[0]))
                tmpstr += f"\n认沽:{put['code'].values[0]}_{put['name'].values[0]}_{put['close'].values[0]:.4f}=itm{int(put['itm'].values[0] * 10000)}+{int(put['otm'].values[0] * 10000)}"# 金额:{int(put['amount'].values[0])}"

            png_dict[key] = tmpstr

        return png_dict, data
        # return png_dict


    def get_option_data(self,png_dict, df_optlist,df_etf):


        periodkey = '1分钟k线'
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

            longtext = f'''认购:{optLongCode}_{longrow['name'].values[0]}_{df_opt['long'].values[-1]:.4f}=itm{long_itm * 10000:.0f}+{long_otm * 10000:.0f}_金额:{(df_long['long'] * df_long['trade']).sum():.0f}万'''
            shorttext = f'''认沽:{optShortCode}_{shortrow['name'].values[0]}_{df_opt['short'].values[-1]:.4f}=itm{short_itm * 10000:.0f}+{short_otm * 10000:.0f}_金额:{(df_short['short'] * df_short['trade']).sum():.0f}万'''
            new_optlist[k] = f'''{longtext}\n{shorttext}'''

        opt_Pivot = df_options.pivot_table(index='datetime', columns='etf', values=['long', 'longm20', 'short', 'shortm20',
                    'Long_crossdw', 'Long_crossup', 'Short_crossdw', 'Short_crossup',
                    'Long_pivotup', 'Long_pivotdw','Short_pivotup', 'Short_pivotdw', 'up',
                    'dw', 'boss', 'bossm10','enterlong', 'entershort'], dropna=False)
        self.new_optlist = new_optlist
        self.data = opt_Pivot