import json, datetime, os,re
import numpy as np
import warnings
import time, requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import configparser

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
# from utils.tdx_hosts import hq_hosts, Exhq_hosts
from utils.tdx_indicator import *
from utils.mytdx_cls import mytdxData
from utils.HollowCandlePlotter_cls import HollowCandlestickPlotter
from utils.playsound import playsound

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
           '&po=1&pz=200&pn=1&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5'+ \
           '&fields=f1,f2,f3,f12,f13,f14,f161,f250,f330,f331,f332,f333,f334,f335,f337,f301,f152&fs=m:10'
    res = requests.get(url1, headers=header)
    tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
    data1 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    data1.rename(columns = field_map0,inplace=True)

    for i in range(2,6,1):
        url1i = url1.replace('&pn=1&',f'&pn={i}&')
        resi = requests.get(url1i, headers=header)
        if len(resi.text) > 500:
            tmpi = re.search(r'^\w+\((.*)\);$', resi.text).group(1).replace('"-"', '"0"')
            data1i = pd.DataFrame(json.loads(tmpi)['data']['diff'])
            data1i.rename(columns=field_map0, inplace=True)
            if len(data1i)>0:
                data1 = pd.concat([data1, data1i])


    url2 = url1[:-1] + '2'
    res = requests.get(url2,headers=header)
    tmp = re.search(r'^\w+\((.*)\);$', res.text).group(1).replace('"-"','"0"')
    data2 = pd.DataFrame(json.loads(tmp)['data']['diff'])
    data2.rename(columns = field_map0,inplace=True)

    for i in range(2,6,1):
        url1i = url2.replace('&pn=1&',f'&pn={i}&')
        resi = requests.get(url1i, headers=header)
        if len(resi.text)>500:
            tmpi = re.search(r'^\w+\((.*)\);$', resi.text).group(1).replace('"-"', '"0"')
            data2i = pd.DataFrame(json.loads(tmpi)['data']['diff'])
            data2i.rename(columns=field_map0, inplace=True)
            if len(data2i)>0:
                data2 = pd.concat([data2, data2i])

    data = pd.concat([data1, data2])
    data = data[list(field_map0.values())]

    data['market'] = data['ETFcode'].apply(lambda x: '沪市' if x[0]=='5' else '深市')
    data['direction'] = data['name'].apply(lambda x: 'call' if '购' in x else 'put')
    data['due_date'] = data['到期日'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d').date())
    data['dte'] = data['due_date'].apply(lambda x: (x-datetime.datetime.now().date()).days)
    data['close'] = data['close'].astype(float)
    data['到期日'] = data['到期日'].astype(str)
    data['行权pct'] = data.apply(lambda x:round(x['行权价']/x['ETFprice']*100-100,2),axis=1)
    data['itm'] = data.apply(lambda x: max(0,x.ETFprice-x['行权价']) if x.direction=='call' else max(0,x['行权价']-x.ETFprice),axis=1)
    data['otm'] = data.apply(lambda x: x.close-x.itm,axis=1)

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
    global dte_high, dte_low, close_Threshold_mean,opt_fn, tdxdata

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
            print(f'{datetime.datetime.now().strftime("%m%d_%H:%M:%S")} reusing option file')
            data = pd.read_csv(opt_fn, encoding='gbk',dtype={'ETFcode':str,'code':str})
        else:
            try:
                data = getAllOptionsV3()
                print(f'{datetime.datetime.now().strftime("%m%d_%H:%M:%S")} New option file')
                data.to_csv(opt_fn, encoding='gbk', index=False, float_format='%.4f')
            except:
                print(f'{datetime.datetime.now().strftime("%m%d_%H:%M:%S")} update failed, reusing option file')
                data = pd.read_csv(opt_fn, encoding='gbk', dtype={'ETFcode': str, 'code': str})
    else:
        print('New option file ' + opt_fn)
        data = getAllOptionsV3()
        data.to_csv(opt_fn,encoding='gbk',index=False, float_format='%.4f')

    data.fillna(0,inplace=True)
    # amtlist = data['amount'].values.tolist()
    # amtlist.sort(ascending=False)
    # amtthreshold = amtlist[200] if len(amtlist)>200 else amtlist[-1]
    data.sort_values(by='amount',ascending=False,inplace=True)
    data['itm'] = data.apply(lambda x: max(0,x.ETFprice-x['行权价']) if x.direction=='call' else max(0,x['行权价']-x.ETFprice),axis=1)
    data['otm'] = data.apply(lambda x: x.close-x.itm,axis=1)

    # data2 = data[data['amount']>amtthreshold]

    png_dict = {}
    for key in etf_dict.keys():
        etfcode = etfcode_dict[key]
        tmpdf = data[(data['ETFcode']==etfcode) & (~data['name'].str.contains('A')) & (data['dte']>dte_low) & (data['dte']<dte_high)]
        tmpdf['tmpfact'] = tmpdf['close'].apply(lambda x: x/close_Threshold_mean if x<=close_Threshold_mean else close_Threshold_mean/x)
        tmpdf['tmpfact2'] = tmpdf['tmpfact']*tmpdf['tmpfact']#*tmpdf['tmpfact']*tmpdf['amount']
        tmpdf.sort_values(by='tmpfact2',ascending=False,inplace=True)
        call = tmpdf[(tmpdf['direction']=='call')][:1]
        put = tmpdf[(tmpdf['direction']=='put')][:1]
        if len(call) == 0:
            tmpstr = f'{key}认购:流动性过滤为空   '
        else:
            tmpstr = '认购:' + call['code'].values[0] + '_' + call['name'].values[0] + '_' + str(
                call['close'].values[0]) + ' =itm' + str(int(call['itm'].values[0]*10000)) + '+' + str(int(call['otm'].values[0]*10000)) + \
                ' 杠杆:'+str(int(call['实际杠杆'].values[0]))
        if len(put) == 0:
            tmpstr += f'\n{key}认沽:流动性过滤为空'
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
    if '成交额' in df_single.columns:
        df_single.rename(columns={'成交额':'amount'},inplace=True)
    df_single.reset_index(drop=True,inplace=True)
    if len(df_single)==0:
        print(f'getSingleCCBData {code} kline error,quitting')
        return
    # df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30') if x[-5:]=='13:00' else x)
    df_single['datetime'] = df_single['datetime'].apply(lambda x: x.replace('13:00','11:30'))

    ccbcode = etf_ccb_dict[name]
    df_ccb =  tdxData.get_kline_data(ccbcode, backset=backset, klines=klines, period=period)
    if len(df_ccb)==0:
        print('getSingleCCBData {code} ccb error, quitting')
        return
    df_ccb.rename(columns={'close':'ccb','high':'ccbh','low':'ccbl','open':'ccbo'},inplace=True)
    data = pd.merge(df_ccb[['datetime','ccb','ccbh','ccbl','ccbo']], df_single[['datetime','open','close','high','low','volume','amount']], on='datetime',how='left')


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

    # for k,v in etf_dict2.items():
    if True:
        k = '中证500'; v = '510500'
        optLongCode = png_dict[k].split('\n')[0].split(':')[1].split('_')[0]
        optShortCode = png_dict[k].split('\n')[1].split(':')[1].split('_')[0]

        ETFprice = tdxdata.get_kline_data(v, backset=backset, klines=klines, period=period)['close'].values[-1]
        longrow = df_optlist.loc[df_optlist['code']==optLongCode]
        shortrow = df_optlist.loc[df_optlist['code']==optShortCode]
        
        df_long = tdxdata.get_kline_data(optLongCode, backset=backset, klines=klines, period=period)
        # print(k,v,optLongCode, 'df_long len:', len(df_long))

        df_long['cp30'] = (df_long['close'] - df_long['close'].rolling(30).min()) / (
                    df_long['close'].rolling(30).max() - df_long['close'].rolling(30).min())
        df_long['cp60'] = (df_long['close'] - df_long['close'].rolling(60).min()) / (
                    df_long['close'].rolling(60,min_periods=31).max() - df_long['close'].rolling(60,min_periods=31).min())
        df_long['cm10'] = df_long['close'].rolling(10).mean()
        df_long['cm20'] = df_long['close'].rolling(20).mean()
        df_long['cabovem10'] = df_long['close'] > df_long['cm10']


        df_long['longm20'] = df_long['close'].rolling(20).mean()

        if len(df_long)<=240:
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
                  (df_long['close'] > df_long['close'].shift(1)), 'Long_pivotup'] =preclose_long
        df_long.loc[(df_long['cp30'] > 0.7) & (df_long['cp60'] > 0.7) & (df_long['cabovem10'] == False) & \
                  (df_long['close'] < df_long['close'].shift(1)), 'Long_pivotdw'] = preclose_long


        df_short = tdxdata.get_kline_data(optShortCode, backset=backset, klines=klines, period=period)
        # print(k, v, optShortCode, 'df_short len:', len(df_short))
        df_short['cp30'] = (df_short['close'] - df_short['close'].rolling(30).min()) / (
                    df_short['close'].rolling(30).max() - df_short['close'].rolling(30).min())
        df_short['cp60'] = (df_short['close'] - df_short['close'].rolling(60).min()) / (
                    df_short['close'].rolling(60,min_periods=31).max() - df_short['close'].rolling(60,min_periods=31).min())
        df_short['cm10'] = df_short['close'].rolling(10).mean()
        df_short['cm20'] = df_short['close'].rolling(20).mean()
        df_short['cabovem10'] = df_short['close'] > df_short['cm10']


        df_short['shortm20'] = df_short['close'].rolling(20).mean()
        if len(df_short)<=240:
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

        df_long.rename(columns={'close':'long'}, inplace=True)
        df_short.rename(columns={'close':'short'}, inplace=True)
        df_opt = pd.merge(df_long[['datetime','long','longm20','Long_crossdw', 'Long_crossup','Long_pivotup','Long_pivotdw']],
                          df_short[['datetime','short','shortm20','Short_crossdw', 'Short_crossup','Short_pivotup','Short_pivotdw']], on='datetime',how='inner')

        long_itm = max(0, ETFprice - longrow['行权价'].values[0])
        long_otm = df_opt['long'].values[-1] - long_itm
        short_itm = max(0, shortrow['行权价'].values[0]-ETFprice)
        short_otm = df_opt['short'].values[-1] - short_itm

        longtext = f'''认购:{optLongCode}_{longrow['name'].values[0]}_{df_opt['long'].values[-1]:.4f}=itm{long_itm*10000:.0f}+{long_otm*10000:.0f}_金额:{(df_long['long']*df_long['trade']).sum():.0f}万'''
        shorttext = f'''认沽:{optShortCode}_{shortrow['name'].values[0]}_{df_opt['short'].values[-1]:.4f}=itm{short_itm*10000:.0f}+{short_otm*10000:.0f}_金额:{(df_short['short']*df_short['trade']).sum():.0f}万'''
        new_optlist = f'''{longtext}\n{shorttext}'''


    return df_opt,longtext,shorttext,new_optlist

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
    df_pivot = df_all.pivot_table(index='datetime',columns='etf',values=['ccb', 'close', 'high', 'low','open','preclose','volume',
               'cm5','cm20','ccp60','pivotup','pivotdw','crossup','crossdw','up','dw','bosssigup','bosssigdw','boss','amount'], dropna=False)

    return df_pivot


def processAll():
    global dp_boss, dp_amount, dp_amtcum,dp_bosspct,dp_preclose,df_optlist, new_optlist,timetitle

    df_day_dp = tdxdata.get_kline_data('999999',0,plotdays+20,9)
    df_day_dp['m5'] = df_day_dp['close'].rolling(5).mean()
    df_day_dp['m20'] = df_day_dp['close'].rolling(20).mean()
    df_day_dp = df_day_dp[20:]
    df_day_dp.reset_index(drop=True, inplace=True)
    df_day_etf = tdxdata.get_kline_data('510500',0,plotdays+20,9)
    df_day_etf['m5'] = df_day_etf['close'].rolling(5).mean()
    df_day_etf['m20'] = df_day_etf['close'].rolling(20).mean()
    df_day_etf = df_day_etf[20:]
    df_day_etf.reset_index(drop=True, inplace=True)
    df_day_ccb = tdxdata.get_kline_data('UC500',0,plotdays+20,9)
    df_day_ccb['m5'] = df_day_ccb['close'].rolling(5).mean()
    df_day_ccb['m20'] = df_day_ccb['close'].rolling(20).mean()
    df_day_ccb = df_day_ccb[20:]
    df_day_ccb.reset_index(drop=True, inplace=True)
    pre_ccb = df_day_ccb.close.iloc[-2]

    df_dapan,dp_preclose = getDPdata()
    df_etf1min = getETFdata()

    df_etf1min.reset_index(drop=False, inplace=True)
    df_etf1min['datetime'] = df_etf1min['datetime'].apply(lambda x: x.replace('13:00','11:30'))

    df_temp = pd.merge(df_dapan, df_etf1min, on='datetime', how='left')

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

    k = '中证500'; v='510500'
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

    # seq = str(len(df_temp)).zfill(3)
    ktime = df_temp['datetime'].values[-1][2:].replace('-','').replace(' ','_')
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    timetitle = f'{ktime}--时间戳 {stamp}'

    dp_boss = df_temp['boss'].ffill().values[-1]/100000000
    dp_amount = df_temp['amttrend'].ffill().values[-1]/100000000
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

    df_opt,longtext,shorttext,new_optlist = getOptiondata()
    df_plot = pd.merge(df_opt, df_temp, on='datetime', how='left')

    boss = df_plot[('boss', k)].values[-1]
    bossr1 = df_plot[('boss', k)].values[-2]
    bossm10 = df_plot[('bossm10', k)].values[-1]
    bossm10r1 = df_plot[('bossm10', k)].values[-2]

    if boss>bossm10 and bossr1<bossm10r1:
        playsound('utils\\morning.mp3')
        print('主力资金上穿均线')
        if pushflag=='Y':
            try:
                msgURL = pushurl + '主力资金上穿均线'
                requests.get(msgURL,timeout=1)
            except:
                pass
    elif boss<bossm10 and bossr1>bossm10r1:
        playsound('utils\\swoosh.mp3')
        print('主力资金下穿均线')
        if pushflag=='Y':
            try:
                msgURL = pushurl + '主力资金下穿均线'
                requests.get(msgURL,timeout=1)
            except:
                pass
    else:
        pass


    if 'index' in df_plot.columns:
        del df_plot['index']
    df_plot.reset_index(drop=True, inplace=True)
    df_plot.reset_index(drop=False, inplace=True)
    # df_plot = df_plot[:220]

    if len(df_plot) < 60:
        maxx = 60
        # fig, axes = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]})
    elif len(df_plot) < 120:
        maxx = 120
        # fig, axes = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]})
    elif len(df_plot) < 180:
        maxx = 180
        # fig, axes = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]})
    else:
        maxx = 241

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]})

    if len(df_plot) <= 120:
        for ax in axes[:,1:]:
            ax[0].set_xticks(np.arange(0, 121, 30))
            ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130'))
    else:
        for ax in axes[:,1:]:
            ax[0].set_xticks(np.arange(0, 241, 30))
            ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500'))


    axes[0][1].hlines(y=dp_preclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
    axes[0][1].plot(df_plot.index, df_plot['close'], linewidth=1, color='red')
    axes[0][1].plot(df_plot.index, df_plot['cm20'], linewidth=0.8, color='red', linestyle='--')
    axes[0][1].plot(df_plot.index, df_plot['avg'], linewidth=1, color='violet')

    ax0b = axes[0][1].twinx()
    ax0c = axes[0][1].twinx()
    ax0d = axes[0][1].twinx()
    ax0e = axes[0][1].twinx()

    ax0b.plot(df_plot.index, df_plot.boss, label='主力资金', color='blue', linewidth=1, alpha=1)
    ax0b.plot(df_plot.index, df_plot.bossm10, color='blue',  linestyle='--', linewidth=0.5,  alpha=1)
    ax0b.set_yticks([])
    ax0c.bar(df_plot.index, df_plot.allamt, label='amount', color='grey', alpha=0.3, zorder=-14)
    ax0c.set_yticks([])
    ax0d.plot(df_plot.index, df_plot.amttrend, label='成交量', color='green', lw=1.5, alpha=0.5)

    ax0e.scatter(df_plot.index, df_plot['pivotup'], label='转折点', marker='^', s=49, c='red', alpha=0.6)
    ax0e.scatter(df_plot.index, df_plot['pivotdw'], label='转折点', marker='v', s=49, c='green', alpha=0.7)
    ax0e.scatter(df_plot.index, df_plot['crossup'], label='底部涨', marker='D', s=25, c='red', alpha=0.7)
    ax0e.scatter(df_plot.index, df_plot['crossdw'], label='顶部跌', marker='D', s=25, c='green', alpha=0.8)
    ax0e.scatter(df_plot.index, df_plot['up'], marker='s', s=9, c='red', alpha=0.3)
    ax0e.scatter(df_plot.index, df_plot['dw'], marker='s', s=9, c='green', alpha=0.3)

    ax0e.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6, linestyle='--', zorder=-25)
    ax0e.set_ylim(-10, 10)
    ax0e.set_yticks([])
    axes[0][1].text(0.5, 1.02, f'大盘资金(蓝线):{dp_boss:.0f}亿 成交量(绿线):{dp_amount:.0f}亿 ',
                 horizontalalignment='center', transform=axes[0][1].transAxes, fontsize=12, fontweight='bold',
                 color='black')

    axes[0][1].minorticks_on()
    axes[0][1].grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
    axes[0][1].grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

    ax0e.legend(loc='upper left', framealpha=0.1)
    ax0b.legend(loc='lower left', framealpha=0.1)

    xa = axes[1][1]
    lastclose = df_plot[('preclose', k)].values[1]
    pct = df_plot[('close',k)].dropna().values[-1] / lastclose*100 - 100
    ccb_pct = df_plot[('ccb', k)].dropna().values[-1] / pre_ccb*100 - 100

    xa.hlines(y=lastclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)

    xa.plot(df_plot.index, df_plot[('close', k)], linewidth=1, linestyle='-', color='red', alpha=1.)
    xa.plot(df_plot.index, df_plot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='--', color='red', alpha=1.)
    xa.plot(df_plot.index, df_plot[f'avg_{k}'], linewidth=1, color='violet')

    xa3 = xa.twinx()
    xa3.scatter(df_plot.index, df_plot[('pivotup', k)], label='转折点', s=25, c='r', marker='^', alpha=0.7, zorder=-10)
    xa3.scatter(df_plot.index, df_plot[('pivotdw', k)], label='转折点', s=25, c='g', marker='v', alpha=0.7, zorder=-10)
    xa3.scatter(df_plot.index, df_plot[('crossup', k)], s=16, c='r', marker='D', alpha=0.7, zorder=-10)
    xa3.scatter(df_plot.index, df_plot[('crossdw', k)], s=16, c='g', marker='D', alpha=0.7, zorder=-10)
    xa3.scatter(df_plot.index, df_plot[('up',k)], marker='s', s=9, c='red', alpha=0.3)
    xa3.scatter(df_plot.index, df_plot[('dw',k)], marker='s', s=9, c='green', alpha=0.3)
    xa3.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6,
              zorder=-25)
    xa3.set_ylim(-10, 10)
    xa3.set_yticks([])

    xa4 = xa.twinx()
    xa4.plot(df_plot.index, df_plot[('ccb', k)], label='ccb',linewidth=0.9, linestyle='-', color='green')
    xa4.hlines(pre_ccb, xmin=df_plot.index.min(), xmax=maxx, color='green', linewidth=0.5, alpha=0.6,
              linestyle='--',zorder=-25)
    xa4.set_yticks([])

    xa5 = xa.twinx()
    xa5.bar(df_plot.index, df_plot[('volume', k)], color='gray', alpha=0.3, zorder=-15)
    xa5.set_yticks([])

    xa6 = xa.twinx()
    xa6.plot(df_plot.index, df_plot[('boss', k)], linewidth=0.8, linestyle='-', color='blue')
    xa6.plot(df_plot.index, df_plot[('bossm10', k)], color='blue', linestyle='--', linewidth=0.5, alpha=1)
    xa.legend(loc='upper left', framealpha=0.1)
    xa3.legend(loc='lower left', framealpha=0.1)

    xa.minorticks_on()
    xa.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
    xa.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

    xa.text(0.7, 1.02, f'{k}涨跌:{pct:.2f}%, ccb涨跌:{ccb_pct:.2f}%',
             horizontalalignment='center', transform=xa.transAxes, fontsize=12, fontweight='bold', color='black')

    xb = axes[2][1]
    longpct = df_plot['long'].values[-1] / df_plot['long'].values[0] * 100 - 100
    shortpct = df_plot['short'].values[-1] / df_plot['short'].values[0] * 100 - 100
    ylim_min = df_plot['long'].dropna().min() * 0.95
    ylim_max = df_plot['long'].dropna().max() * 1.05


    xb.plot(df_plot.index, df_plot['long'], linewidth=0.6, label='认购(左)', linestyle='-', color='red')
    xb.plot(df_plot.index, df_plot['longm20'], linewidth=0.6, linestyle='--', color='red')
    xb.hlines(y=df_plot['long'].dropna().iloc[0], xmin=df_plot.index.min(), linestyle='--', xmax=maxx,
             colors='red', lw=1, alpha=0.5, zorder=-20)
    xb.scatter(df_plot.index, df_plot['Long_crossup'], marker='o', s=16, color='red', alpha=0.5, zorder=-10)
    xb.scatter(df_plot.index, df_plot['Long_crossdw'], marker='o', s=16, color='green', alpha=0.5, zorder=-10)
    xb.scatter(df_plot.index, df_plot['Long_pivotup'], marker='^', s=16, color='red', alpha=0.5, zorder=-10)
    xb.scatter(df_plot.index, df_plot['Long_pivotdw'], marker='v', s=16, color='green', alpha=0.5, zorder=-10)
    xb.scatter(df_plot.index, df_plot['up'], marker='s', s=9, color='red', alpha=0.3, zorder=-10)
    xb.scatter(df_plot.index, df_plot['dw'], marker='s', s=9, color='green', alpha=0.3, zorder=-10)
    xb.set_ylim(ylim_min, ylim_max)

    xb1 = xb.twinx()
    xb1.plot(df_plot.index, df_plot['short'], linewidth=0.6, label='认沽(右)', linestyle='-', color='green')
    xb1.plot(df_plot.index, df_plot['shortm20'], linewidth=0.6, linestyle='--', color='green')
    xb1.hlines(y=df_plot['short'].dropna().iloc[0], xmin=df_plot.index.min(), linestyle='--', xmax=maxx,
              colors='green', lw=1, alpha=0.5, zorder=-20)
    xb1.scatter(df_plot.index, df_plot['Short_crossup'], marker='o', s=16, color='red', alpha=0.5, zorder=-10)
    xb1.scatter(df_plot.index, df_plot['Short_crossdw'], marker='o', s=16, color='green', alpha=0.5, zorder=-10)

    xb1.scatter(df_plot.index, df_plot['Short_pivotup'], marker='^', s=16, color='red', alpha=0.5, zorder=-10)
    xb1.scatter(df_plot.index, df_plot['Short_pivotdw'], marker='v', s=16, color='green', alpha=0.5, zorder=-10)

    xb2 = xb.twinx()
    xb2.plot(df_plot.index, df_plot[('boss', k)], linewidth=0.8, linestyle='-', color='blue')
    xb2.plot(df_plot.index, df_plot[('bossm10', k)], color='blue', linestyle='--', linewidth=0.5, alpha=1)
    xb2.set_yticks([])

    xb.text(0.25, 0.95, new_optlist, horizontalalignment='center', transform=xb.transAxes, fontsize=12,
               fontweight='bold', color='black')
    xb.text(0.9, 1.02, f'认购:{longpct:.0f}%  认沽:{shortpct:.0f}%', horizontalalignment='center',
           transform=xb.transAxes, fontsize=12,
           fontweight='bold', color='black')

    xb.minorticks_on()
    xb.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
    xb.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

    xa4.legend(loc='upper right', framealpha=0.1)
    xb.legend(loc='center left', fontsize=10, frameon=True, framealpha=0.1)
    xb1.legend(loc='center right', fontsize=10, frameon=True, framealpha=0.1)

    collections0 = plotter.construct_collections(df_day_dp)
    for collection in collections0:
        axes[0][0].add_collection(collection)
    axes[0][0].plot(df_day_dp.index, df_day_dp.m5, linewidth=0.8, linestyle='--', color='red')
    axes[0][0].plot(df_day_dp.index, df_day_dp.m20, linewidth=0.8, linestyle='-.', color='red')

    collections1 = plotter.construct_collections(df_day_etf)
    for collection in collections1:
        axes[1][0].add_collection(collection)
    axes[1][0].plot(df_day_etf.index, df_day_etf.m5, linewidth=0.8, linestyle='--', color='red')
    axes[1][0].plot(df_day_etf.index, df_day_etf.m20, linewidth=0.8, linestyle='-.', color='red')

    collections2 = plotter.construct_collections(df_day_ccb)
    for collection in collections2:
        axes[2][0].add_collection(collection)
    axes[2][0].plot(df_day_ccb.index, df_day_ccb.m5, linewidth=0.8, linestyle='--', color='green')
    axes[2][0].plot(df_day_ccb.index, df_day_ccb.m20, linewidth=0.8, linestyle='-.', color='green')

    axes[0][0].text(0.5, 1.02, f'大盘指数日线  {timetitle}',
                 horizontalalignment='center', transform=axes[0][0].transAxes, fontsize=12, fontweight='bold',
                 color='black')
    axes[1][0].text(0.5, 1.02, f'中证500日线',
                 horizontalalignment='center', transform=axes[1][0].transAxes, fontsize=12, fontweight='bold',
                 color='black')
    axes[2][0].text(0.5, 1.02, f'中证500持仓比',
                 horizontalalignment='center', transform=axes[2][0].transAxes, fontsize=12, fontweight='bold',
                 color='black')

    axes[0][0].set_xticks(np.arange(0, len(df_day_dp), 5))
    xtick_labels = [i[2:10].replace('-', '') for i in df_day_dp['datetime'].iloc[::5]]
    axes[0][0].set_xticklabels(xtick_labels, rotation=45)
    axes[1][0].set_xticks(np.arange(0, len(df_day_etf), 5))
    xtick_labels = [i[2:10].replace('-', '') for i in df_day_etf['datetime'].iloc[::5]]
    axes[1][0].set_xticklabels(xtick_labels, rotation=45)
    # axes[1][0].set_xticklabels([])
    axes[2][0].set_xticks(np.arange(0, len(df_day_ccb), 5))
    xtick_labels = [i[2:10].replace('-', '') for i in df_day_ccb['datetime'].iloc[::5]]
    axes[2][0].set_xticklabels(xtick_labels, rotation=45)
    axes[0][0].grid(True)
    axes[1][0].grid(True)
    axes[2][0].grid(True)


    plt.tight_layout()
    plt.savefig(f'output\\持续监控全景_v2.9_{datetime.datetime.now().strftime("%Y%m%d")}.png')

    fig.clf()
    plt.close(fig)

    return df_temp

def main():

    global factor, dayr1,png_dict, tdxdata, df_optlist,df_full,opt_fine,hq_hosts,Exhq_hosts

    if (time.strftime("%H%M", time.localtime()) > '0900' and time.strftime("%H%M", time.localtime()) <= '0930'):
        print(f'waiting market, sleep {sleepsec*2}s')
        time.sleep(sleepsec*2)

    try:

        while (time.strftime("%H%M", time.localtime())>='0920' and time.strftime("%H%M", time.localtime())<='1502'):

            if (time.strftime("%H%M", time.localtime())>'1131' and time.strftime("%H%M", time.localtime())<'1300'):
                print(f'sleep {sleepsec*2}s')
                time.sleep(sleepsec*2)
            elif (time.strftime("%H%M", time.localtime())<'0930'):
                print(f'sleep {sleepsec*2}s')
                time.sleep(sleepsec*2)
            else:
                try:
                    png_dict, df_optlist = getMyOptions()
                except:
                    png_dict = {'k': '流动性'}
                opt_fine = True

                processAll()
                time.sleep(sleepsec)

        processAll()

        return
    except Exception as e:
        print('exception msg: '+ str(e))
        print(' *****  exception, recreate tdxdata then restart main ***** ')
        tdxdata.api.close()
        tdxdata.Exapi.close()
        tdxdata = mytdxData(hq_hosts,Exhq_hosts)
        time.sleep(10)
        main()


if __name__ == '__main__':

    prog_start = time.time()
    print('-------------------------------------------')
    print('Job start !!! ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))

    cfg_fn = 'monitor_v2.9.cfg'
    config = configparser.ConfigParser()
    config.read(cfg_fn, encoding='utf-8')
    dte_low = int(dict(config.items('option_screen'))['dte_low'])
    dte_high = int(dict(config.items('option_screen'))['dte_high'])
    close_Threshold_mean = float(dict(config.items('option_screen'))['close_mean'])
    etf_ccb_dict = dict(config.items('etf_ccb_dict'))
    etfbk_dict = dict(config.items('etfbk_dict'))
    etf_dict = dict(config.items('etf_dict'))
    etf_dict2 = dict(config.items('etf_dict2'))
    etfcode_dict = dict(config.items('etfcode_dict'))
    kline_dict = dict(config.items('kline_dict'))
    kline_qty = dict(config.items('kline_qty'))
    backset = int(dict(config.items('backset'))['backset'])
    sleepsec = int(dict(config.items('sleep'))['seconds'])
    png_dict = dict(config.items('png_dict'))
    opt_path = dict(config.items('path'))['opt_path']
    pushurl = dict(config.items('pushmessage'))['url']
    pushflag = dict(config.items('pushmessage'))['enable']
    plotopt = dict(config.items('plotimgs'))['option']
    plotdays = int(dict(config.items('plotimgs'))['days'])

    # 解析 tdx_hosts
    hq_hosts = []
    for key in config['tdx_hosts']:
        name, ip, port = config['tdx_hosts'][key].split(',')
        hq_hosts.append((name.strip(), ip.strip(), int(port.strip())))

    # 解析 tdx_exhosts
    Exhq_hosts = []
    for key in config['tdx_exhosts']:
        name, ip, port = config['tdx_exhosts'][key].split(',')
        Exhq_hosts.append((name.strip(), ip.strip(), int(port.strip())))

    marketcolors = {
        'candle': {'up': 'r', 'down': 'g'},
        'edge': {'up': 'r', 'down': 'g'},
        'wick': {'up': 'r', 'down': 'g'},
        'hollow': (1.0, 1.0, 1.0, 0.0),
        'alpha': 0.9
    }

    MPLconfig = {'_width_config': {'candle_width': 0.4, 'candle_linewidth': 0.8}}
    plotter = HollowCandlestickPlotter(marketcolors=marketcolors, config=MPLconfig)

    tdxdata  = mytdxData(hq_hosts,Exhq_hosts)

    now = tdxdata.get_kline_data('399001',0,20,8)
    tempdates = tdxdata.get_kline_data('399001',0,20,9)
    opt_fn_last =  opt_path +  '\\沪深期权清单_'+ tempdates['datetime'].values[-2][:10].replace('-','')+'.csv'
    opt_fn =  opt_path +  '\\沪深期权清单_'+ now['datetime'].values[-1][:10].replace('-','')+'.csv'
    print(opt_fn)

    try:
        png_dict,df_optlist = getMyOptions()
    except Exception as e:
        print(f'获取期权标的失败：{e}')
        png_dict = {'k':'流动性'}
    opt_fine = True

    factor = calAmtFactor(5)
    factor = factor+[1.00]

    # processAll()

    main()

    tdxdata.api.close()
    tdxdata.Exapi.close()

    time_end = time.time()
    print('-------------------------------------------')
    print(f'Job completed!!!  All time costed: {(time_end - prog_start):.0f}秒')
