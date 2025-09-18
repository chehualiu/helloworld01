# monitor.py
import time
from option_monitor import OptionMonitor
import warnings
warnings.filterwarnings("ignore")


def main():
    start_time = time.time()
    print("Job start !!!", time.strftime('%Y%m%d_%H:%M:%S'))

    monitor = OptionMonitor(config_file='monitor_v3.0.cfg')

    try:
        monitor.run()
    except KeyboardInterrupt:
        print(f'{time.strftime("%H:%M:%S", time.localtime())} Monitoring stopped by user.')
    finally:
        monitor.tdx_data.close()
        end_time = time.time()
        print(f'Job completed in {(end_time - start_time):.0f} seconds.')


if __name__ == '__main__':

    # monitor = OptionMonitor(config_file='monitor_v3.0.cfg')
    # png_dict, df_optlist = monitor.options_fetcher.get_my_options(monitor.config['option_screen'])
    # # 校验完整性
    # valid = all('流动性' not in v for v in png_dict.values())
    # if not valid:
    #     print("期权未能选取完整，请检查配置")
    # tt = monitor.etf_fetcher.get_full_data()
    # if len(tt) > 0:
    #     monitor.plotter.plot_full_day(monitor.etf_fetcher, png_dict)
    #
    #     if monitor.config['plotimgs']['option'] == 'Y' and valid:
    #         monitor.options_fetcher.get_option_data(png_dict, df_optlist, monitor.etf_fetcher.data)
    #         monitor.plotter.plot_options(png_dict, monitor.options_fetcher.data, monitor.options_fetcher.new_optlist)
    #     time.sleep(monitor.sleepsec)

    main()
