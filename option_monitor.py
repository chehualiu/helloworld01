# option_monitor.py
import pandas as pd
import time
from utils.config_loader import load_config
from data.options_fetcher import OptionsDataFetcher
from data.etf_fetcher import ETFDataFetcher
from data.market_plotter import MarketPlotter
from utils.mytdx_cls import mytdxData


class OptionMonitor:
    def __init__(self, config_file='monitor_v2.8.cfg'):
        self.config = load_config(config_file)
        hosts = self.config['tdx_hosts']
        ex_hosts = self.config['tdx_exhosts']
        self.tdx_data = mytdxData(hosts, ex_hosts)
        self.options_fetcher = OptionsDataFetcher(self.tdx_data, self.config['etf_dict'], self.config['etf_dict2'], self.config['path']['opt_path'])
        self.etf_fetcher = ETFDataFetcher(self.tdx_data, self.config['etf_dict2'], self.config['etf_ccb_dict'], self.config['etfbk_dict'])
        self.plotter = MarketPlotter(self.config['etf_dict'])
        self.sleepsec = self.config['sleep_seconds']

    def run(self):
        while (time.strftime("%H%M", time.localtime())>='0910' and time.strftime("%H%M", time.localtime())<='1502'):

            if (time.strftime("%H%M", time.localtime())>'1131' and time.strftime("%H%M", time.localtime())<'1300'):
                print(f'{time.strftime("%H:%M:%S", time.localtime())} sleep {self.sleepsec*2}s')
                time.sleep(self.sleepsec*2)
            elif (time.strftime("%H%M", time.localtime())<'0930'):
                print(f'{time.strftime("%H:%M:%S", time.localtime())} sleep {self.sleepsec*2}s')
                time.sleep(self.sleepsec*2)
            else:
                try:
                    png_dict, df_optlist = self.options_fetcher.get_my_options(self.config['option_screen'])
                    # 校验完整性
                    valid = all('流动性' not in v for v in png_dict.values())
                    if not valid:
                        print("期权未能选取完整，请检查配置")
                    self.etf_fetcher.get_full_data()
                    self.plotter.plot_full_day(self.etf_fetcher, png_dict)
                    if self.config['plotimgs']['option'] == 'Y' and valid:
                        self.options_fetcher.get_option_data(png_dict, df_optlist, self.etf_fetcher.data)
                        self.plotter.plot_options(png_dict, self.options_fetcher.data, self.options_fetcher.new_optlist)
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
        self.tdx_data.close()
        hosts = self.config['tdx_hosts']
        ex_hosts = self.config['tdx_exhosts']
        self.tdx_data = mytdxData(hosts, ex_hosts)
        time.sleep(10)
