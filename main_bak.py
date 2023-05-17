import threading
from flask import Flask, send_file, render_template, jsonify

import json, datetime, os,re
import numpy as np
import warnings
import time, requests
import pandas as pd

import matplotlib.pyplot as plt

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
import utils.tdxExhq_config as conf

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# pd.describe_option()
pd.options.mode.chained_assignment = None
pd.set_option('precision', 6)
pd.set_option('display.precision', 4)
pd.set_option('display.float_format',  '{:.2f}'.format)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('expand_frame_repr', False)
warnings.filterwarnings('ignore')
np.random.seed(42)


app = Flask(__name__)

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
    min['northdeltam5'] = min['northdelta'].rolling(5).mean()

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

def MINgetDPindex():
    global AmtFactor

    url = 'http://push2his.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery112409887318613050171_1615514818501&secid=1.000001&secid2=0.399001'+ \
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
    factor = [AmtFactor[0]] + AmtFactor

    for i in range(len(data)):
        data['amttrend'][i] = data['amtcum'][i] * factor[i]
    data.drop(columns=['amtcum','volcum','vol','amt'], inplace=True)

    data['close'] = data['close'].astype('float')
    data['avg'] = data['avg'].astype('float')

    # data = get_higher_low_up(data, 3)

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
        data1 = json.loads(res.text[42:-2])['data']['trends']
        preclose = json.loads(res.text[42:-2])['data']['preClose']
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

def min2day(timestr):
    global AmtFactor
    factor = AmtFactor
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
    amt_all = df_min['amount'].sum()
    df_result = pd.DataFrame(data=df_min['time'].values[0:240], columns=['minute'])
    df_result['minpct'] = 0
    for min in df_min['time'].values[0:240]:
        amt_min = df_min[df_min['time']==min]['amount'].sum()
        df_result.loc[df_result['minute']==min, 'minpct'] = amt_min / amt_all
    df_result['pct'] = df_result['minpct'].cumsum()
    df_result['factor'] = 1/df_result['pct']

    return list(df_result['factor'].values)

def plotAllzjlx():
    df_north = getNorthzjlx()  # northdelta
    hs300_zjlx = getHS300zjlx()  # zjlxdelta
    hs300_index,hs_preclose = getETFindex('510300')  #   close
    if len(hs300_index) == 0:
        print('failed to get 300ETF df, skipped')
        return
    dp_zjlx = MINgetZjlxDP()   # zjlxdelta
    dp_index, dp_preclose = MINgetDPindex()  # close

    dp_boss = dp_zjlx.boss.values[-1]/100000000
    dp_north = df_north.north.values[-1]
    hs300_boss = hs300_zjlx.boss.values[-1]
    upcnt = hs300_index.upcnt.values[-2]
    df_dp = pd.merge(df_north, dp_zjlx, left_index=True, right_index=True)
    df_dp = pd.merge(df_dp, dp_index, left_index=True, right_index=True)
    df_hs300 = pd.merge(hs300_zjlx, hs300_index,left_index=True, right_index=True)
    dp_amount = str(round(list(df_dp.amttrend)[-1]/100000000,2))

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
        print('UP 510300 hs300' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = 'http://wx.xtuis.cn/XfhmghWTRzitW6RHalZc8AzN5.send?text=' + msg
        requests.get(msgURL)
    if df_hs300['dw'].values[-1]==True:
        msg = 'UP 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))
        print('DOWN 510300 hs300' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_hs300['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
        msgURL = 'http://wx.xtuis.cn/XfhmghWTRzitW6RHalZc8AzN5.send?text=' + msg
        requests.get(msgURL)
    df_hs300['up'] = df_hs300.apply(lambda x: x.close if x.up==True else np.nan, axis=1)
    df_hs300['dw'] = df_hs300.apply(lambda x: x.close if x.dw==True else np.nan, axis=1)

    if len(df_hs300)<240:
        df_hs300 = pd.concat([df_hs300, pd.DataFrame([[]]*(240-len(df_hs300)))])
    df_hs300.reset_index(drop=True,inplace=True)
    df_hs300.reset_index(inplace=True)


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
        print('UP 999999 沪市大盘' + ' 低位上穿ma10 -- '+  ' close '+  str(round(df_dp['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
    if df_dp['dw'].values[-1]==True:
        print('DOWN 999999 沪市大盘' + ' 高位下穿ma10 -- '+  ' close '+  str(round(df_dp['close'].values[-1],3))+ \
                       '  止损价:' + '-----' + ' 止损%:' +  ' ---- %')
    df_dp['up'] = df_dp.apply(lambda x: x.close if x.up==True else np.nan, axis=1)
    df_dp['dw'] = df_dp.apply(lambda x: x.close if x.dw==True else np.nan, axis=1)
    if len(df_dp)<240:
        df_dp = pd.concat([df_dp, pd.DataFrame([[]]*(240-len(df_dp)))])
    df_dp.reset_index(drop=True,inplace=True)
    df_dp.reset_index(inplace=True)

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
    ax1c.plot(df_dp.index, df_dp.northdeltam5, label='northdeltam5', color='black', lw=0.4)
    ax1c.hlines(y=0, xmin=0, xmax=240-3, colors='black', linestyles='-', lw=0.3)
    ax1d.plot(df_dp.index, df_dp.amttrend, label='amttrend', color='green', lw=1.5, alpha=0.5)
    ax1.text(0.5,0.92,' 大盘主力资金(蓝):' + str(round(dp_boss,2)) + ' 北向流入(灰):' + str(round(dp_north,2)) + '  成交量(绿)' ,
             horizontalalignment='center',transform=ax1.transAxes, fontsize=12, fontweight='bold', color='black')

    df_hs300.plot(x='index', y='close', label='hs300', linewidth=1, color='red', ax=ax2,zorder=10)
    df_hs300.plot(x='index', y='avg', linewidth=1, markersize=10, color='violet', label='avg', ax=ax2,zorder=11)
    ax2.hlines(y=hs_preclose, xmin=0, xmax=240-3, colors='aqua', linestyles='-', lw=2, label='preclose')
    ax2.scatter(df_hs300.index, df_hs300['up'], marker='^', s=100, c='red',alpha=0.7)
    ax2.scatter(df_hs300.index, df_hs300['dw'], marker='v',s=100, c='green',alpha=0.7)
    ax2b.bar(df_hs300.index, df_hs300.net, label='zjlx', color='blue', alpha=0.2, zorder=-15)
    ax2b.plot(df_hs300.index, df_hs300.bossma5, label='zjlxm5', color='blue', lw=0.5)
    ax2c.plot(df_hs300.index, df_hs300.upcnt, label='upcnt', color='green', lw=1.5,alpha=0.5 )
    ax2.text(0.5,0.9,'HS300 主力流入:' + str(round(hs300_boss,2)) + ' 市场上涨家数: '+ str(round(upcnt,0)),
             horizontalalignment='center',transform=ax2.transAxes, fontsize=12, fontweight='bold', color='black')
    ax1b.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax1d.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)
    ax2c.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False)

    ax1.minorticks_on()
    ax1.grid(b=True, which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    # ax1.grid(b=True, which='minor', axis="both", color='k', linestyle='dotted', linewidth=0.5)
    ax1.set(xlabel=None)

    ax2.minorticks_on()
    ax2.grid(b=True, which='major', axis="both", color='k', linestyle='-', linewidth=0.5)
    # ax2.grid(b=True, which='minor', axis="both", color='k', linestyle='dotted', linewidth=0.5)
    ax2.set(xlabel=None)

    plt.suptitle('DP HS300 - 时间戳 ' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
    plt.tight_layout()
    plt.savefig('static/image.png')
    fig.clf()
    plt.close(fig)
    return

def main():

    global api, Exapi, AmtFactor, dayr1

    try:
        api.close()
        Exapi.close()
    except:
        time.sleep(10)

    try:
        api = TdxHq_API(heartbeat=True)
        Exapi = TdxExHq_API(heartbeat=True)

        if TestConnection(api, 'HQ', conf.HQsvr, conf.HQsvrport )==False:
            print('connection to TDX server not available')
        if TestConnection(Exapi, 'ExHQ', conf.ExHQsvr, conf.ExHQsvrport )==False:
            print('connection to Ex TDX server not available')

        AmtFactor = calAmtFactor(5)
        plotAllzjlx()

        while (time.strftime("%H%M", time.localtime())>='0930' and time.strftime("%H%M", time.localtime())<='2002'):
            if (time.strftime("%H%M", time.localtime())>'1130' and time.strftime("%H%M", time.localtime())<'1300'):
                print('sleep 60s')
                time.sleep(60)
            else:
                plotAllzjlx()
                print('img refreshed!')
                time.sleep(30)

        api.close()
        Exapi.close()
        return
    except Exception as e: # work on python 3.x
        print('exception msg: '+ str(e))
        print(' *****  exception, restart main ***** ')
        time.sleep(5)
        main()

def image_generation_thread():
    global generation_thread
    main()


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':

    generation_thread = threading.Thread(target=image_generation_thread)
    generation_thread.start()
    generation_running = True

    app.run()

