# option_monitor.py
import pandas as pd
import time,requests
from utils.config_loader import load_config
from data.options_fetcher import OptionsDataFetcher
from data.etf_fetcher import ETFDataFetcher
from data.market_plotter import MarketPlotter
from utils.mytdx_cls import mytdxData
from utils.zjlx_min_maintain_20251015 import get_BK_Zjlx
from utils.get_cookie import get_eastmoney_cookie


class OptionMonitor:
    def __init__(self, config_file='monitor_v3.0.cfg'):
        self.config = load_config(config_file)
        hosts = self.config['tdx_hosts']
        ex_hosts = self.config['tdx_exhosts']
        self.trade_start = self.config['trade_start']
        self.trade_end = self.config['trade_end']
        self.tdx_data = mytdxData(hosts, ex_hosts, speed_test=True)
        self.options_fetcher = OptionsDataFetcher(self.tdx_data, self.config['etf_dict'], self.config['etf_dict2'], self.config['path']['opt_path'])
        self.etf_fetcher = ETFDataFetcher(self.tdx_data, self.config['etf_dict2'], self.config['etf_ccb_dict'], self.config['etfbk_dict'])
        self.plotter = MarketPlotter(self.config['etf_dict'])
        self.sleepsec = self.config['sleep_seconds']
        self.cookie = self.config['cookie']
        self.data = pd.DataFrame()
        self.pushurl = self.config['pushmessage']['url']
        self.cookieUpdateTime = None


    def is_trading_time(self):
        now = time.strftime("%H%M", time.localtime())
        return self.trade_start <= now <= self.trade_end

    def is_pre_market(self):
        now = time.strftime("%H%M", time.localtime())
        return '0900' <= now <= '0930'
    def is_noon_break(self):
        now = time.strftime("%H%M", time.localtime())
        return '1131' <= now <= '1259'

    def update_zjlx_data(self, cookie, emzjlx_frequency):

        tmp_dict = self.config['etfbk_dict'].copy()
        tmp_dict['dapan'] = 'dapan'

        for bk, bkcode in tmp_dict.items():
            timetiklast = self.etf_fetcher.bkzjlx_time_dict[bk]
            if timetiklast is None or (time.time() - timetiklast) > emzjlx_frequency:
                timetik, df_dpzjlx_tmp = get_BK_Zjlx(bkcode, cookie)

                should_update_cookie = self.cookieUpdateTime is None or (
                            time.time() - self.cookieUpdateTime) > 60
                if len(df_dpzjlx_tmp) == 0 and should_update_cookie:
                    tmp = get_eastmoney_cookie()
                    if len(tmp)>10:
                        self.cookie = tmp
                        self.cookieUpdateTime = time.time()
                        print(f'{time.strftime("%H:%M:%S", time.localtime())} cookie updated')
                    else:
                        print(f'{time.strftime("%H:%M:%S", time.localtime())} cookie update failed')

                if len(df_dpzjlx_tmp) > 0:
                    self.etf_fetcher.bkzjlx_time_dict[bk] = timetik
                    self.etf_fetcher.bkzjlx_data_dict[bk] = df_dpzjlx_tmp
                elif len(self.etf_fetcher.bkzjlx_data_dict[bk]) > 0 and self.etf_fetcher.bkzjlx_data_dict[bk]['datetime'].values[-1] > '2020-01-01':
                    print(f'{time.strftime("%H:%M:%S", time.localtime())} {bk} update failed, use last version')
                elif len(self.etf_fetcher.bkzjlx_data_dict[bk]) == 0:
                    print(f'{time.strftime("%H:%M:%S", time.localtime())} {bk} update failed, use fake data')
                    self.etf_fetcher.bkzjlx_data_dict[bk] = pd.DataFrame({'datetime': ['2000-01-01'], 'boss': [0]})
                else:
                    print(f'{time.strftime("%H:%M:%S", time.localtime())} dapan unexpected branch')
                    self.etf_fetcher.bkzjlx_data_dict[bk] = pd.DataFrame({'datetime': ['2000-01-01'], 'boss': [0]})
            else:
                pass  # no change

    def run(self, first_run=False):
        while self.is_trading_time() or first_run:

            if self.is_noon_break():
                print(f'{time.strftime("%H:%M:%S", time.localtime())} 中午休息 sleep {self.sleepsec*2}s')
                time.sleep(self.sleepsec*2)
            elif self.is_pre_market():
                print(f'{time.strftime("%H:%M:%S", time.localtime())} 盘前竞价 {self.sleepsec*2}s')
                time.sleep(self.sleepsec*2)
            else:
                try:
                    self.update_zjlx_data(self.cookie, 50)
                    png_dict, df_optlist = self.options_fetcher.get_my_options(self.config['option_screen'], cookie=self.cookie)
                    # 校验完整性
                    valid = all('流动性' not in v for v in png_dict.values())
                    if not valid:
                        print("期权未能选取完整，请检查配置")
                    tt = self.etf_fetcher.get_full_data()
                    self.check_dp_boss(tt)

                    if len(tt)>0:
                        self.plotter.plot_full_day(self.etf_fetcher, png_dict)

                        if self.config['plotimgs']['option'] == 'Y' and valid:
                            self.options_fetcher.get_option_data(png_dict, df_optlist, self.etf_fetcher.data)
                            self.plotter.plot_options(png_dict, self.options_fetcher.data, self.options_fetcher.new_optlist)
                        time.sleep(self.sleepsec)
                    else:
                        print(f"{time.strftime('%H:%M:%S', time.localtime())} 数据获取失败")
                        time.sleep(self.sleepsec)
                    first_run = False
                except Exception as e:
                    print(f"Error occurred: {e}")
                    self.restart_connection()
                    time.sleep(10)


    def restart_connection(self):
        try:
            self.tdx_data.reconnect()
        finally:
            time.sleep(5)

    def check_dp_boss(self, dd):
        # 主力资金上穿10分钟均线，上涨信号
        if dd['boss'].values[-1] > dd['bossm10'].values[-1] and dd['boss'].values[-2] < \
                dd['bossm10'].values[-2]:
            msg = '大盘主力资金上穿10分钟均线，1min上涨信号'
            msgURL = f'{self.pushurl}1min上涨信号&desp={msg}'
            requests.get(msgURL)
        elif dd['boss'].values[-1] < dd['bossm10'].values[-1] and dd['boss'].values[-2] > \
                dd['bossm10'].values[-2]:
            msg = '大盘主力资金下穿10分钟均线，1min下跌信号'
            msgURL = f'{self.pushurl}1min下跌信号&desp={msg}'
            requests.get(msgURL)

