import requests, json, struct
import numpy as np
import pandas as pd

import json, datetime, os, sys
import warnings
import time, math, requests
# from .EM_zjlx import get_BK_Zjlx
from tenacity import retry, stop_after_attempt, wait_fixed

# pd.options.mode.chained_assignment = None
# pd.set_option('display.max_rows',100)
# pd.set_option('display.max_columns',40)
# # pd.set_option('display.width',1000)
# pd.set_option('display.precision', 5)
# warnings.filterwarnings('ignore')
# np.random.seed(42)

@retry(stop=stop_after_attempt(1), wait=wait_fixed(5))
def safe_get_request(url, headers=None, params=None):
    return requests.get(url, headers=headers, params=params, proxies={})

def get_BK_Zjlx(bkcode, cookie):

    bkzjlx_url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        # 'Cookie': 'qgqp_b_id=2e5f72b5a8e430f5f578199b44b2c650; st_nvi=QTo1DhRK-9b0WLzl763GF90fb; nid=005c4e7016dc8547c46bc58c842215af; nid_create_time=1754291547115; gvi=KrcKF3koRrZAdqQD855wDc6b3; gvi_create_time=1754291547115; mtp=1; ct=cI7Lo5aNpHXvsN7fQyYXistdSwQypZ6pCRwLMoxfH0iERvj4LVcqu917dRLv7P67zYRdhnod-iTNFzaJa_kSIEtIDFUZn_1C8_NHoD_Y5RKRBCYJYOZwT5neykaLga0PXwdy35PQZGq_4CFY3lRy-_qbkh7idXA7TLSHoOJ-TPo; ut=FobyicMgeV63FIqRjcfDeRRI5bofMxGbDM6MboJ-5VcfKNE1KSeQpn0nTjA4MCb02RzK21f9bRjUYAzm70AllVVVIdtMEotGI_s9g10b0UaKP69Thj2H5QcD1ZiZSO-OEilTSeDyI7ovhg9WIMkOWNAaNzej3GSP1SsySJd2uveX0gCto16y4-lYWnQCYc7ToqxcAYY3c-nDrc7cseHQ_o8fLJc2yUu_KeKuEZfiOMyUYm0sP5ZFQPNbs-8sWRqxoZq2gmtlQV0x3BUHmu1HgVrSbeyYDla2; pi=2186316006663558%3Bu2186316006663558%3B%E8%82%A1%E5%8F%8BDOk6GW%3BYNLvxdi0J24eaLlSY08HJwyb5HP5SUwF2j7%2FqoTM2s6oi6XGG3xHh3NoXD44aZOh%2Fq%2BdgQuAKtcnwn95MBNB7pvGwFHdKbbmxLeeKSK6KPN%2Bg%2FzZz6CFR71Kdfy45FQE3JCvNVotzQnXQbfXHrjG4wMjYpLYsPktYSGL6ycrDQdESRNzfOEgfq3iunRh6Y8ZHfoh%2BXCG%3BgzA1ItCCoCphbnEKpXwGOtx2wcjvzwYLQ0gL6%2B9lwUMiY5IPkp%2BloqrpW6WGzsIscBnu8lrcdhaidpNjazeWLFjyNXpQwjzkheO9Ds8%2FuI3lwnEiMf3ay%2BffRVKbX0OmFM5iiXwsTn6iEPTV3ik2mK29yhMoeg%3D%3D; uidal=2186316006663558%e8%82%a1%e5%8f%8bDOk6GW; sid=152379750; vtpst=|; fullscreengg=1; fullscreengg2=1; st_si=15442846996330; st_asi=delete; st_pvi=99659171718134; st_sp=2020-12-25%2019%3A43%3A29; st_inirUrl=http%3A%2F%2Ffinance.eastmoney.com%2Fa%2F202002261396627810.html; st_sn=10; st_psi=20251014132624338-113300300992-9960592741',
        'Cookie': cookie,
        'Host': 'push2.eastmoney.com',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'sec-ch-ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }

    timetick = str(int(time.time() * 1000))
    params = {
        'cb': f'jQuery112305705962762541327_{timetick}',
        'lmt': '0',
        'klt': '1',
        'fields1': 'f1,f2,f3,f7',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65',
        'ut': 'b2884a393a59ad64002292a3e90d46a5',
        'secid': f'90.{bkcode}',
        '_': timetick  # '1760419583577'
    }

    if bkcode == 'dapan':
        del params['secid'], params['_']
        params['secid'] = "1.000001"
        params['secid2'] = "0.399001"
        params['_'] = timetick
        params['cb'] = f"jQuery112307580484684912303_{timetick}"

    headers['cookie'] = cookie
    # url = bkzjlx_url.replace('BK0500',bkcode)
    # res = requests.get(bkzjlx_url, params=params, headers=headers,proxies={})
    # query_string = urlencode(params, doseq=True)
    # real_url = f"{bkzjlx_url}?{query_string}"
    # print(f'{bkcode} URL: {real_url}')

    try:
        # res = safe_get_request(bkzjlx_url, headers=headers, params=params)
        res = requests.get(bkzjlx_url, params=params, headers=headers,proxies={})
        # res = self.safe_get_request(base_url, params=params)
        data1 = json.loads(res.text[res.text.index('{'):-2])['data']['klines']
    except:
        print(f'get_BK_Zjlx {bkcode} data error')
        return time.time(), pd.DataFrame()
    min = pd.DataFrame([i.split(',') for i in data1], columns=['datetime', 'boss', 'small', 'med', 'big', 'huge'])
    min.drop(labels=['small', 'med', 'big', 'huge'], axis=1, inplace=True)
    # min['time'] = min['time'].astype('datetime64[ns]')
    min['boss'] = min['boss'].astype('float') / 100000000

    return time.time(), min


# def getStockZjlxRT(code):
#
#     if code=='399001':
#         url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=0.399001&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery18309083856080027362_1608023040666&_=1608023043234'
#     elif code=='999999':
#         url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=1.000001&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery18300286863114494722_1608022996738&_=1608022997258'
#     elif code[:1]=='6':
#         url='http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=1.' + code + '&'+ \
#             'fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&' + \
#             'ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery18305259214827586713_1607786227748&_=1607786227891'
#     else:
#         url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=0.' + code + '&'+ \
#               'fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&' + \
#               'ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery18305259214827586713_1607786227748&_=1607786227891'
#
#     res = requests.get(url)
#     try:
#         data1 = json.loads(res.text[41:-2])['data']['klines']
#     except:
#         return pd.DataFrame()
#     min = pd.DataFrame([i.split(',') for i in data1],columns=['time', 'boss', 'small', 'med', 'big', 'huge'])
#     min.drop(labels=['small','med','big','huge'],axis=1,inplace=True)
#     min['time'] = min['time'].astype('datetime64[ns]')
#     min['boss'] = min['boss'].astype('float')/10000
#     min['delta'] = min['boss'] - min['boss'].shift(1)
#     min['ma5'] = min['delta'].rolling(5).mean()
#     return min


# def getBKZjlxRT(bkcode):
#     url = 'http://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=90.' + bkcode + \
#           '&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56&ut=fa5fd1943c7b386f172d6893dbfba10b&cb=jQuery112406142175621622367_1615545163205&_=1615545163206'
#     res = requests.get(url)
#
#     try:
#         data1 = json.loads(res.text[42:-2])['data']['klines']
#     except:
#         return pd.DataFrame()
#     min = pd.DataFrame([i.split(',') for i in data1],columns=['time', 'boss', 'small', 'med', 'big', 'huge'])
#     min.drop(labels=['small','med','big','huge'],axis=1,inplace=True)
#     # min['time'] = min['time'].astype('datetime64[ns]')
#     min['boss'] = min['boss'].astype('float')/100000000
#
#     return min

def saveZJLXdata(cookie=None):

    today = datetime.datetime.now().strftime('%Y%m%d')
    data_dir = r"D:\stockProg\win2023\output\zjlx"

    BK_list = {'hsdapan':'dapan', '510300':'BK0500', '中证_510500':'BK0701', '创业_159915':'BK0638','科创50_588000':'BK1108'}
    for name,bkcode in BK_list.items():
        # df_bk = getBKZjlxRT(bkcode)
        _, df_bk = get_BK_Zjlx(bkcode, cookie=cookie)
        if len(df_bk)<200:
            continue
        fn = data_dir+'\\zjlx_min_'+ name + '.csv'
        if os.path.exists(fn):
            try:
                temp = pd.read_csv(fn, encoding='gbk')
                lasttime = temp.time.values[-1]
                if 'datetime' in df_bk.columns:
                    df_bk.rename(columns={'datetime':'time'},inplace=True)
                if len(df_bk[df_bk['time']>lasttime])==0:
                    continue
                else:
                    data = pd.concat([temp, df_bk[df_bk['time']>lasttime]])
                    data[['time','boss']].to_csv(fn,index=False,encoding='gbk',float_format='%.4f')
            except:
                print(fn, 'can not be opened')
                df_bk[['time', 'boss']].to_csv(fn, index=False, encoding='gbk', float_format='%.4f')
        else:
            if len(df_bk)>0:
                df_bk.to_csv(fn,index=False,encoding='gbk',float_format='%.4f')

    print('data refresh completed!')

if __name__ == '__main__':

    print('Job Started !!!', datetime.datetime.now(), '\n')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    prog_start = time.time()

    cookie = 'qgqp_b_id=1296411ee76ff7730dfcad6866f999b1; st_nvi=9tPzGRrddxABSsl2PgP1n272a; st_si=23335772256745; nid=0fb1d9c49b03318096b53bba0088ecd4; nid_create_time=1760509594771; gvi=waQTWnkGHmybhqeOUOulY2d86; gvi_create_time=1760509594771; fullscreengg=1; fullscreengg2=1; st_asi=delete; wsc_checkuser_ok=1; st_pvi=99659171718134; st_sp=2020-12-25%2019%3A43%3A29; st_inirUrl=http%3A%2F%2Ffinance.eastmoney.com%2Fa%2F202002261396627810.html; st_sn=30; st_psi=20251017151249621-113300300871-7030113081'
    saveZJLXdata(cookie= cookie)

    prog_end = time.time()
    print('\n\nAll time costed: ', round(prog_end - prog_start, 2))
    print('Job Completed !!!', datetime.datetime.now(), '\n')
