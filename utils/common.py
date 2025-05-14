# utils/common.py
import pandas as pd
import numpy as np
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 全局变量初始化（改为属性注入）
pd.options.mode.chained_assignment = None
pd.set_option('display.precision', 4)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

# 创建输出目录
if not os.path.exists('output'):
    os.makedirs('output')
