# plotting/market_plotter.py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


class MarketPlotter:

    def __init__(self, etf_dict: dict):
        self.etf_dict = etf_dict
        # self.dp_preclose = dp_preclose
        # self.etf_bk_dict = bk_dict
    def plot_full_day(self, etf_fetcher, png_dict):

        df = etf_fetcher.data

        ktime = df['datetime'].values[-1][2:].replace('-','').replace(' ','_')
        stamp = datetime.now().strftime('%H:%M:%S')
        timetitle = f'{ktime}--时间戳 {stamp}'

        if len(df) <= 121:
            df_plot = pd.concat([df, pd.DataFrame([[]] * (121 - len(df)))])
        elif len(df) > 121 and len(df) <= 241:
            df_plot = pd.concat([df, pd.DataFrame([[]] * (241 - len(df)))])
        else:
            print('df>240')
            return

        df_plot.reset_index(drop=True, inplace=True)
        df_plot.reset_index(drop=False, inplace=True)
        datalen = len(df_plot['datetime'].dropna())
        if datalen < 60:
            maxx = 60
        elif datalen < 120:
            maxx = 120
        elif datalen < 180:
            maxx = 180
        else:
            maxx = 240

        # print(plt.get_backend())  tkagg
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))  # 修改为 3 行 2 列布局

        # 设置 x 轴刻度标签
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks(np.arange(0, 241, 30))
                ax.set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500'))

        # 第一个子图 (0, 0)
        axes[0][0].hlines(y=etf_fetcher.dp_preclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
        axes[0][0].plot(df_plot.index, df_plot['close'], linewidth=1, color='red')
        axes[0][0].plot(df_plot.index, df_plot['cm20'], linewidth=0.8, color='red', linestyle='--')
        axes[0][0].plot(df_plot.index, df_plot['avg'], linewidth=1, color='violet')

        ax0c = axes[0][0].twinx()
        ax0d = axes[0][0].twinx()
        ax0e = axes[0][0].twinx()

        ax0c.bar(df_plot.index, df_plot.allamt, label='amount', color='grey', alpha=0.3, zorder=-14)
        ax0c.set_yticks([])
        # ax0d.plot(df_plot.index, df_plot.amttrend, label='成交量', color='green', lw=1.5, alpha=0.5)
        ax0d.plot(df_plot.index, df_plot.boss, label='主力资金', color='blue', linewidth=1, alpha=1)
        ax0d.plot(df_plot.index, df_plot.bossm10, color='blue', linestyle='--', linewidth=0.5, alpha=1)

        # ax0e.scatter(df_plot.index, df_plot['pivotup'], label='转折点', marker='^', s=49, c='red', alpha=0.6)
        # ax0e.scatter(df_plot.index, df_plot['pivotdw'], label='转折点', marker='v', s=49, c='green', alpha=0.7)
        # ax0e.scatter(df_plot.index, df_plot['crossup'], label='底部涨', marker='D', s=16, c='red', alpha=0.6)
        # ax0e.scatter(df_plot.index, df_plot['crossdw'], label='顶部跌', marker='D', s=16, c='green', alpha=0.7)
        ax0e.scatter(df_plot.index, df_plot['up'], marker='s', s=9, c='red', alpha=0.3)
        ax0e.scatter(df_plot.index, df_plot['dw'], marker='s', s=9, c='green', alpha=0.3)

        ax0e.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6, linestyle='--',
                    zorder=-25)
        ax0e.set_ylim(-10, 10)
        ax0e.set_yticks([])

        axes[0][0].text(0.5, 1.02, f'主力(蓝线):{etf_fetcher.bkzjlx_data_dict["dapan"]["boss"].dropna().values[-1]:.0f}亿   {timetitle}',
                        horizontalalignment='center', transform=axes[0][0].transAxes, fontsize=12, fontweight='bold',
                        color='black')

        funcax0 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x, (x / etf_fetcher.dp_preclose - 1) * 100)
        axes[0][0].yaxis.set_major_formatter(mtick.FuncFormatter(funcax0))

        axes[0][0].minorticks_on()
        axes[0][0].grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        axes[0][0].grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

        # ax0e.legend(loc='upper left', framealpha=0.1)

        keylist = list(self.etf_dict.keys())
        funcx0 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[0])].values[1] - 1) * 100)
        funcx1 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[1])].values[1] - 1) * 100)
        funcx2 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[2])].values[1] - 1) * 100)
        funcx3 = lambda x, pos: "{:.3f}\n{:.1f}%".format(x, (x / df_plot[('preclose', keylist[3])].values[1] - 1) * 100)

        func_zdjs = lambda x, pos: "{:.0f}\n{:.0f}%".format(x, (x / df_plot['zdjs'].values[0] - 1) * 100)
        axes[0][1].plot(df_plot.index, df_plot['zdjs'], label='上涨家数', linewidth=1, color='red')
        axes[0][1].hlines(df_plot['zdjs'].min(), xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5,
                          alpha=0.0, zorder=-25)
        axes[0][1].yaxis.set_major_formatter(mtick.FuncFormatter(func_zdjs))
        axes[0][1].minorticks_on()
        axes[0][1].grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        axes[0][1].grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)
        axes[0][1].legend(loc='upper left', framealpha=0.1)
        axes[0][1].text(0.5, 1.02, f'成交量(绿线):{etf_fetcher.dp_amount:.0f}亿  上涨家数:{etf_fetcher.zdjs:.0f}',
                        horizontalalignment='center', transform=axes[0][1].transAxes, fontsize=12, fontweight='bold',
                        color='black')

        ax0b1 = axes[0][1].twinx()
        # ax0b.plot(df_plot.index, df_plot.boss, label='主力资金', color='blue', linewidth=1, alpha=1)
        # ax0b.plot(df_plot.index, df_plot.bossm10, color='blue', linestyle='--', linewidth=0.5, alpha=1)
        ax0b1.plot(df_plot.index, df_plot.amttrend, label='成交量', color='green', lw=1.5, alpha=0.5)
        # ax0b.set_yticks([])
        # axes[0][1].legend(loc='upper left', framealpha=0.1)
        ax0b1.legend(loc='upper right', framealpha=0.1)

        # 第二行：两个子图 (1, 0) 和 (1, 1)
        for i, k in enumerate(self.etf_dict.keys()):
            if i == 0:
                x = axes[1][0]
            elif i == 1:
                x = axes[1][1]
            elif i == 2:
                x = axes[2][0]
            elif i == 3:
                x = axes[2][1]
            else:
                continue

            lastclose = df_plot[('preclose', k)].values[1]
            pct = df_plot[('close', k)].dropna().values[-1] / lastclose * 100 - 100

            x.hlines(y=lastclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
            x.plot(df_plot.index, df_plot[('close', k)], linewidth=1, linestyle='-', color='red', alpha=1.)
            x.plot(df_plot.index, df_plot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='--', color='red',
                   alpha=1.)
            x.plot(df_plot.index, df_plot[f'avg_{k}'], linewidth=1, color='violet')
            x.scatter(df_plot.index, df_plot[f'c_enterlong_{k}'], marker='o', s=9, c='red', alpha=0.5)
            x.scatter(df_plot.index, df_plot[f'c_entershort_{k}'], marker='o', s=9, c='green', alpha=0.7)

            x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx0 if i == 0 else funcx1))

            x3 = x.twinx()
            # x3.scatter(df_plot.index, df_plot[('pivotup', k)], label='转折点', s=25, c='r', marker='^', alpha=0.7,
            #            zorder=-10)
            # x3.scatter(df_plot.index, df_plot[('pivotdw', k)], label='转折点', s=25, c='g', marker='v', alpha=0.7,
            #            zorder=-10)
            # x3.scatter(df_plot.index, df_plot[('crossup', k)], s=16, c='r', marker='D', alpha=0.6, zorder=-10)
            # x3.scatter(df_plot.index, df_plot[('crossdw', k)], s=16, c='g', marker='D', alpha=0.7, zorder=-10)
            x3.scatter(df_plot.index, df_plot[('up', k)], marker='s', s=9, c='red', alpha=0.3)
            x3.scatter(df_plot.index, df_plot[('dw', k)], marker='s', s=9, c='green', alpha=0.3)
            x3.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6, zorder=-25)
            x3.set_ylim(-10, 10)
            x3.set_yticks([])

            x4 = x.twinx()
            x4.plot(df_plot.index, df_plot[('ccb', k)], linewidth=0.9, linestyle='-', color='green')
            x4.plot(df_plot.index, df_plot[('ccbm20', k)], linewidth=0.9, linestyle='--', color='green')
            x4.set_yticks([])

            x5 = x.twinx()
            x5.bar(df_plot.index, df_plot[('volume', k)], color='gray', alpha=0.3, zorder=-15)
            x5.set_yticks([])

            x6 = x.twinx()
            x6.plot(df_plot.index, df_plot[('boss', k)], linewidth=0.8, linestyle='-', color='blue')
            x6.plot(df_plot.index, df_plot[('bossm10', k)], color='blue', linestyle='--', linewidth=0.5, alpha=1)

            # x.legend(loc='upper left', framealpha=0.1)
            # x3.legend(loc='lower left', framealpha=0.1)

            x.minorticks_on()
            x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
            x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

            if k in png_dict.keys():
                x.text(0.5, 1.03, png_dict[k], horizontalalignment='center', transform=x.transAxes, fontsize=12,
                       fontweight='bold', color='black')
            ccb_key = f"{k}_CP"
            ccb_percentile = etf_fetcher.ccb_range[ccb_key] * 100
            x.text(0.5, 0.94, f'{k} 涨跌:{pct:.2f}%  CCB涨跌:{etf_fetcher.ccb_pctChg[k]*100:.2f}%_百分位:{ccb_percentile:.0f}%',
                   horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        # # 第三行：两个子图 (2, 0) 和 (2, 1)
        # for i, k in enumerate(self.etf_dict.keys()):
        #     if i == 2:
        #         x = axes[2][0]
        #     elif i == 3:
        #         x = axes[2][1]
        #     else:
        #         continue
        #
        #     lastclose = df_plot[('preclose', k)].values[1]
        #     pct = df_plot[('close', k)].dropna().values[-1] / lastclose * 100 - 100
        #
        #     x.hlines(y=lastclose, xmin=df_plot.index.min(), xmax=maxx, colors='aqua', linestyles='-', lw=2)
        #     x.plot(df_plot.index, df_plot[('close', k)], linewidth=1, linestyle='-', color='red', alpha=1.)
        #     x.plot(df_plot.index, df_plot[('cm20', k)], label='ma20', linewidth=0.7, linestyle='--', color='red',
        #            alpha=1.)
        #     x.plot(df_plot.index, df_plot[f'avg_{k}'], linewidth=1, color='violet')
        #     x.scatter(df_plot.index, df_plot[f'c_enterlong_{k}'], marker='o', s=9, c='red', alpha=0.5)
        #     x.scatter(df_plot.index, df_plot[f'c_entershort_{k}'], marker='o', s=9, c='green', alpha=0.7)
        #
        #     x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx2 if i == 2 else funcx3))
        #
        #     x3 = x.twinx()
        #     x3.scatter(df_plot.index, df_plot[('pivotup', k)], label='转折点', s=25, c='r', marker='^', alpha=0.7,
        #                zorder=-10)
        #     x3.scatter(df_plot.index, df_plot[('pivotdw', k)], label='转折点', s=25, c='g', marker='v', alpha=0.7,
        #                zorder=-10)
        #     x3.scatter(df_plot.index, df_plot[('crossup', k)], s=16, c='r', marker='D', alpha=0.6, zorder=-10)
        #     x3.scatter(df_plot.index, df_plot[('crossdw', k)], s=16, c='g', marker='D', alpha=0.7, zorder=-10)
        #     x3.scatter(df_plot.index, df_plot[('up', k)], marker='s', s=9, c='red', alpha=0.3)
        #     x3.scatter(df_plot.index, df_plot[('dw', k)], marker='s', s=9, c='green', alpha=0.3)
        #     x3.hlines(0, xmin=df_plot.index.min(), xmax=maxx, color='k', linewidth=0.5, alpha=0.6, zorder=-25)
        #     x3.set_ylim(-10, 10)
        #     x3.set_yticks([])
        #
        #     x4 = x.twinx()
        #     x4.plot(df_plot.index, df_plot[('ccb', k)], linewidth=0.9, linestyle='-', color='green')
        #     x4.plot(df_plot.index, df_plot[('ccbm20', k)], linewidth=0.9, linestyle='--', color='green')
        #     x4.set_yticks([])
        #
        #     x5 = x.twinx()
        #     x5.bar(df_plot.index, df_plot[('volume', k)], color='gray', alpha=0.3, zorder=-15)
        #     x5.set_yticks([])
        #
        #     x6 = x.twinx()
        #     x6.plot(df_plot.index, df_plot[('boss', k)], linewidth=0.8, linestyle='-', color='blue')
        #     x6.plot(df_plot.index, df_plot[('bossm10', k)], color='blue', linestyle='--', linewidth=0.5, alpha=1)
        #
        #     # x.legend(loc='upper left', framealpha=0.1)
        #     x3.legend(loc='lower left', framealpha=0.1)
        #
        #     x.minorticks_on()
        #     x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
        #     x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)
        #
        #     if k in png_dict.keys():
        #         x.text(0.5, 1.03, png_dict[k], horizontalalignment='center', transform=x.transAxes, fontsize=12,
        #                fontweight='bold', color='black')
        #     ccb_key = f"{k}_CP"
        #     ccb_percentile = etf_fetcher.ccb_range[ccb_key] * 100
        #     x.text(0.5, 0.94, f'{k} 涨跌:{pct:.2f}%   CCB涨跌:{etf_fetcher.ccb_pctChg[k]*100:.1f}%_百分位:{ccb_percentile:.0f}%',
        #            horizontalalignment='center', transform=x.transAxes, fontsize=12, fontweight='bold', color='black')

        plt.tight_layout()
        plt.savefig(f'output\\持续监控全景_v3.0_{datetime.now().strftime("%Y%m%d")}.png')
        fig.clf()
        plt.close()

    def plot_options(self, png_dict, df_opt: pd.DataFrame,new_optlist:dict):

        # df_opt = getOptiondata()
        # df_opt = df_opt#[:100]

        df_opt.reset_index(drop=False, inplace=True)
        if 'index' in df_opt.columns:
            del df_opt['index']
        df_opt.reset_index(drop=True, inplace=True)
        df_opt.reset_index(drop=False, inplace=True)

        keylist = list(self.etf_dict.keys())
        funcx00 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000,
                                                          (x / df_opt[('long', keylist[0])].dropna().iloc[0] - 1) * 100)
        funcx01 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000, (
                    x / df_opt[('short', keylist[0])].dropna().iloc[0] - 1) * 100)
        funcx10 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000,
                                                          (x / df_opt[('long', keylist[1])].dropna().iloc[0] - 1) * 100)
        funcx11 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000, (
                    x / df_opt[('short', keylist[1])].dropna().iloc[0] - 1) * 100)
        funcx20 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000,
                                                          (x / df_opt[('long', keylist[2])].dropna().iloc[0] - 1) * 100)
        funcx21 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000, (
                    x / df_opt[('short', keylist[2])].dropna().iloc[0] - 1) * 100)
        funcx30 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000,
                                                          (x / df_opt[('long', keylist[3])].dropna().iloc[0] - 1) * 100)
        funcx31 = lambda x, pos: "{:.0f}\n{:.1f}%".format(x * 10000, (
                    x / df_opt[('short', keylist[3])].dropna().iloc[0] - 1) * 100)

        timestamp = df_opt['datetime'].values[-1].replace('-', '')

        data_len = len(df_opt)

        if data_len < 30:
            maxx = 60
        elif data_len < 60:
            maxx = 90
        elif data_len < 120:
            maxx = 120
        elif data_len < 180:
            maxx = 180
        else:
            maxx = 240
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax in axes[:, :1]:
            ax[0].set_xticks(np.arange(0, 241, 30))
            # ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130'))
            ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500'))
        for ax in axes[:, 1:]:
            ax[0].set_xticks(np.arange(0, 241, 30))
            ax[0].set_xticklabels(('930', '1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500'))

        for i, k in enumerate(self.etf_dict.keys()):
            if i == 0:
                x = axes[0][0]
            elif i == 1:
                x = axes[0][1]
            elif i == 2:
                x = axes[1][0]
            else:
                x = axes[1][1]

            longpct = df_opt[('long', k)].values[-1] / df_opt[('long', k)].values[0] * 100 - 100
            shortpct = df_opt[('short', k)].values[-1] / df_opt[('short', k)].values[0] * 100 - 100
            ylim_min = df_opt[('long', k)].dropna().min() * 0.95
            ylim_max = df_opt[('long', k)].dropna().max() * 1.05
            # df_opt[('bossm10', k)] = df_opt[('boss', k)].rolling(10).mean()

            x.plot(df_opt.index, df_opt[('long', k)], linewidth=0.6, label='认购(左)', linestyle='-', color='red')
            x.plot(df_opt.index, df_opt[('longm20', k)], linewidth=0.6, linestyle='--', color='red')
            x.hlines(y=df_opt[('long', k)].dropna().iloc[0], xmin=df_opt.index.min(), linestyle='--', xmax=maxx,
                     colors='red', lw=1, alpha=0.5, zorder=-20)
            # x.scatter(df_opt.index, df_opt[('Long_crossup', k)], marker='o', s=16, color='red', alpha=0.5, zorder=-10)
            # x.scatter(df_opt.index, df_opt[('Long_crossdw', k)], marker='o', s=16, color='green', alpha=0.5, zorder=-10)
            # x.scatter(df_opt.index, df_opt[('Long_pivotup', k)], marker='^', s=16, color='red', alpha=0.5, zorder=-10)
            # x.scatter(df_opt.index, df_opt[('Long_pivotdw', k)], marker='v', s=16, color='green', alpha=0.5, zorder=-10)
            # x.scatter(df_opt.index, df_opt[('up', k)], marker='s', s=9, color='red', alpha=0.3, zorder=-10)
            # x.scatter(df_opt.index, df_opt[('dw', k)], marker='s', s=9, color='green', alpha=0.3, zorder=-10)
            x.scatter(df_opt.index, df_opt[('enterlong', k)], marker='o', s=9, color='red', alpha=0.5, zorder=-10)
            x.scatter(df_opt.index, df_opt[('entershort', k)], marker='o', s=9, color='green', alpha=0.6, zorder=-10)
            x.set_ylim(ylim_min, ylim_max)

            x1 = x.twinx()
            x1.plot(df_opt.index, df_opt[('short', k)], linewidth=0.6, label='认沽(右)', linestyle='-', color='green')
            x1.plot(df_opt.index, df_opt[('shortm20', k)], linewidth=0.6, linestyle='--', color='green')
            x1.hlines(y=df_opt[('short', k)].dropna().iloc[0], xmin=df_opt.index.min(), linestyle='--', xmax=maxx,
                      colors='green', lw=1, alpha=0.5, zorder=-20)
            # x1.scatter(df_opt.index, df_opt[('Short_crossup', k)], marker='o', s=16, color='red', alpha=0.5, zorder=-10)
            # x1.scatter(df_opt.index, df_opt[('Short_crossdw', k)], marker='o', s=16, color='green', alpha=0.5, zorder=-10)
            # x1.scatter(df_opt.index, df_opt[('Short_pivotup', k)], marker='^', s=16, color='red', alpha=0.5, zorder=-10)
            # x1.scatter(df_opt.index, df_opt[('Short_pivotdw', k)], marker='v', s=16, color='green', alpha=0.5,
            #            zorder=-10)

            x2 = x.twinx()
            x2.plot(df_opt.index, df_opt[('boss', k)], label='主力资金', color='blue', linewidth=0.7, alpha=1)
            x2.plot(df_opt.index, df_opt[('bossm10', k)], color='blue', linestyle='--', linewidth=0.5, alpha=1)
            x2.set_yticks([])

            x3 = x.twinx()
            x3.scatter(df_opt.index, df_opt[('up', k)], marker='s', s=9, color='red', alpha=0.3, zorder=-10)
            x3.scatter(df_opt.index, df_opt[('dw', k)], marker='s', s=9, color='green', alpha=0.3, zorder=-10)
            x3.scatter(df_opt[df_opt[('Long_crossup', k)]>0].index, [1.1]*len(df_opt[df_opt[('Long_crossup', k)]>0]), marker='^', s=36, color='red', alpha=0.5, zorder=-10)
            x3.scatter(df_opt[df_opt[('Long_crossdw', k)]>0].index, [1.1]*len(df_opt[df_opt[('Long_crossdw', k)]>0]), marker='v', s=36, color='green', alpha=0.5, zorder=-10)
            x3.scatter(df_opt[df_opt[('Short_pivotup', k)]>0].index, [0.9]*len(df_opt[df_opt[('Short_pivotup', k)]>0]), marker='v', s=36, color='green', alpha=0.5, zorder=-10)
            x3.scatter(df_opt[df_opt[('Short_pivotdw', k)]>0].index, [0.9]*len(df_opt[df_opt[('Short_pivotdw', k)]>0]), marker='^', s=36, color='red', alpha=0.5, zorder=-10)
            x3.set_ylim(-1, 3.0)
            x3.set_yticks([])

            if k in png_dict.keys():
                x.text(0.5, 1.02, new_optlist[k], horizontalalignment='center', transform=x.transAxes, fontsize=12,
                       fontweight='bold', color='black')
            x.text(0.5, 0.96, f'认购:{longpct:.0f}%  认沽:{shortpct:.0f}%', horizontalalignment='center',
                   transform=x.transAxes, fontsize=12,
                   fontweight='bold', color='black')

            if i == 0:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx00))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx01))
            elif i == 1:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx10))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx11))
            elif i == 2:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx20))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx21))
            else:
                x.yaxis.set_major_formatter(mtick.FuncFormatter(funcx30))
                x1.yaxis.set_major_formatter(mtick.FuncFormatter(funcx31))

            x.minorticks_on()
            x.grid(which='major', axis="both", color='k', linestyle='--', linewidth=0.3)
            x.grid(which='minor', axis="x", color='k', linestyle='dotted', linewidth=0.15)

            x.legend(loc='center left', fontsize=10, frameon=True, framealpha=0.1)
            x1.legend(loc='center right', fontsize=10, frameon=True, framealpha=0.1)

        plt.tight_layout()
        ktime = df_opt['datetime'].values[-1][2:].replace('-','').replace(' ','_')
        stamp = datetime.now().strftime('%H:%M:%S')
        timetitle = f'{ktime}--时间戳 {stamp}'
        plt.suptitle(timetitle, x=0.5, y=1.0)
        plt.savefig(f'output\\持续监控_期权_v3.0_{datetime.now().strftime("%Y%m%d")}.png')

        fig.clf()

        plt.close()
