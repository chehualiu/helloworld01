# option_monitor.py
import pandas as pd
import time
from utils.config_loader import load_config
from data.options_fetcher import OptionsDataFetcher
from data.etf_fetcher import ETFDataFetcher
from data.market_plotter import MarketPlotter
from utils.mytdx_cls import mytdxData


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

    def is_trading_time(self):
        now = time.strftime("%H%M", time.localtime())
        return self.trade_start <= now <= self.trade_end

    def is_pre_market(self):
        now = time.strftime("%H%M", time.localtime())
        return '0900' <= now <= '0930'
    def is_noon_break(self):
        now = time.strftime("%H%M", time.localtime())
        return '1131' <= now <= '1259'
    def run(self):
        while self.is_trading_time():

            if self.is_noon_break():
                print(f'{time.strftime("%H:%M:%S", time.localtime())} 中午休息 sleep {self.sleepsec*2}s')
                time.sleep(self.sleepsec*2)
            elif self.is_pre_market():
                print(f'{time.strftime("%H:%M:%S", time.localtime())} 盘前竞价 {self.sleepsec*2}s')
                time.sleep(self.sleepsec*2)
            else:
                try:
                    png_dict, df_optlist = self.options_fetcher.get_my_options(self.config['option_screen'], cookie=self.cookie)
                    # 校验完整性
                    valid = all('流动性' not in v for v in png_dict.values())
                    if not valid:
                        print("期权未能选取完整，请检查配置")
                    tt = self.etf_fetcher.get_full_data()
                    if len(tt)>0:
                        self.plotter.plot_full_day(self.etf_fetcher, png_dict)

                        if self.config['plotimgs']['option'] == 'Y' and valid:
                            self.options_fetcher.get_option_data(png_dict, df_optlist, self.etf_fetcher.data)
                            self.plotter.plot_options(png_dict, self.options_fetcher.data, self.options_fetcher.new_optlist)
                        time.sleep(self.sleepsec)
                    else:
                        print(f"{time.strftime('%H:%M:%S', time.localtime())} 数据获取失败")
                        time.sleep(self.sleepsec)
                except Exception as e:
                    print(f"Error occurred: {e}")
                    self.restart_connection()

        try:
            png_dict, df_optlist = self.options_fetcher.get_my_options(self.config['option_screen'])
            # 校验完整性
            valid = all('流动性' not in v for v in png_dict.values())
            if not valid:
                print("期权未能选取完整，请检查配置")
                return
            self.etf_fetcher.get_full_data()
            self.plotter.plot_full_day(self.etf_fetcher, png_dict)
            if self.config['plotimgs']['option'] == 'Y' and valid:
                self.options_fetcher.get_option_data(png_dict, df_optlist, self.etf_fetcher.data)
                self.plotter.plot_options(png_dict, self.options_fetcher.data, self.options_fetcher.new_optlist)
            time.sleep(self.config['sleep_seconds'])
        except Exception as e:
            print(f"Error occurred: {e}")
            return


    def restart_connection(self):
        try:
            self.tdx_data.reconnect()
        finally:
            # self.tdx_data = mytdxData(self.config['tdx_hosts'], self.config['tdx_exhosts'])
            time.sleep(5)
