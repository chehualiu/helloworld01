# utils/config_loader.py
import configparser
import os


def load_config(config_file='monitor_v2.8.cfg'):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")

    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    def _parse_dict(section):
        return dict(config.items(section))

    parsed = {
        'option_screen': dict(config.items('option_screen')),
        'etf_dict': dict(config.items('etf_dict')),
        'etf_dict2': dict(config.items('etf_dict2')),
        'etfcode_dict': dict(config.items('etfcode_dict')),
        'kline_dict': dict(config.items('kline_dict')),
        'kline_qty': dict(config.items('kline_qty')),
        'backset': int(dict(config.items('backset'))['backset']),
        'sleep_seconds': int(dict(config.items('sleep'))['seconds']),
        'png_dict': dict(config.items('png_dict')),
        'path': dict(config.items('path')),
        'pushmessage': dict(config.items('pushmessage')),
        'plotimgs': dict(config.items('plotimgs')),
        'etfbk_dict': dict(config.items('etfbk_dict')),
        'etf_ccb_dict': dict(config.items('etf_ccb_dict')),
        'tdx_hosts': [
            (name.strip(), ip.strip(), int(port.strip()))
            for key in config['tdx_hosts']
            for name, ip, port in [config['tdx_hosts'][key].split(',')]
        ],
        'tdx_exhosts': [
            (name.strip(), ip.strip(), int(port.strip()))
            for key in config['tdx_exhosts']
            for name, ip, port in [config['tdx_exhosts'][key].split(',')]
        ]
    }

    return parsed
