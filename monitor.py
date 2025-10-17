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
        monitor.run(first_run=True)
    except KeyboardInterrupt:
        print(f'{time.strftime("%H:%M:%S", time.localtime())} Monitoring stopped by user.')
    finally:
        monitor.tdx_data.close()
        end_time = time.time()
        print(f'Job completed in {(end_time - start_time):.0f} seconds.')


if __name__ == '__main__':

    main()
