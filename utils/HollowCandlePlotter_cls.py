import numpy as np
# import pandas as pd
import matplotlib.collections as mcol
import matplotlib.colors as mcolors


class HollowCandlestickPlotter:
    """
    用于绘制空心K线（Hollow Candlestick）的类。
    """

    def __init__(self, marketcolors=None, config=None):
        """
        初始化绘图器。

        :param marketcolors: 市场颜色配置，包含 up、down、hollow 等颜色定义。
        :param config: 配置参数，如 alpha 透明度等。
        """
        self.marketc0lors = marketcolors or {
            'candle': {'up': 'r', 'down': 'g'},
            'edge': {'up': 'r', 'down': 'g'},
            'wick': {'up': 'r', 'down': 'g'},
            'hollow': (1.0, 1.0, 1.0, 0.0),  # 默认透明
            'alpha': 0.9
        }
        self.config = config or {'_width_config': {'candle_width': 0.4, 'candle_linewidth': 0.8}}

    def _updownhollow_colors(self, upcolor, downcolor, hollowcolor, opens, closes):
        """
        根据开盘价和收盘价生成空心/实心K线体的颜色列表。
        """
        if upcolor == downcolor:
            return [upcolor] * len(opens)
        umap = {True: hollowcolor, False: upcolor}
        dmap = {True: hollowcolor, False: downcolor}
        return [umap[opn < cls] if opn < cls else dmap[opn < cls] for opn, cls in zip(opens, closes)]

    def _updown_colors(self, upcolor, downcolor, opens, closes, use_prev_close=False):
        """
        根据价格变动方向返回颜色列表（上升或下降）。
        """
        if upcolor == downcolor:
            return [upcolor] * len(opens)
        cmap = {True: upcolor, False: downcolor}
        if not use_prev_close:
            return [cmap[opn < cls] for opn, cls in zip(opens, closes)]
        else:
            first = cmap[opens[0] < closes[0]]
            _list = [cmap[pre < cls] for cls, pre in zip(closes[1:], closes[:-1])]
            return [first] + _list

    def _mpf_to_rgba(self, c, alpha=None):
        """
        将颜色转换为 RGBA 格式。
        """
        if isinstance(c, tuple) and any(e > 1 for e in c[:3]):
            c = tuple([e / 255. for e in c[:3]] + list(c[3:] if len(c) == 4 else []))
        return mcolors.to_rgba(c, alpha)

    def construct_collections(self, data):
        """
        构建 K 线图的集合对象（LineCollection 和 PolyCollection）。

        :param data: 包含 OHLC 数据的 DataFrame。
        :return: 返回 LineCollection 和 PolyCollection 对象。
        """
        opens = data['open'].values
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values

        indexes = np.arange(len(data))
        delta = 0.4  # 可以根据 config 调整宽度

        barVerts = [((date - delta, open),
                     (date - delta, close),
                     (date + delta, close),
                     (date + delta, open)) for date, open, close in zip(indexes, opens, closes)]

        rangeSegLow = [((date, low), (date, min(open, close))) for date, low, open, close in
                       zip(indexes, lows, opens, closes)]
        rangeSegHigh = [((date, high), (date, max(open, close))) for date, high, open, close in
                        zip(indexes, highs, opens, closes)]
        rangeSegments = rangeSegLow + rangeSegHigh

        alpha = self.marketc0lors.get('alpha', 0.9)
        uc = self._mpf_to_rgba(self.marketc0lors['candle']['up'], alpha)
        dc = self._mpf_to_rgba(self.marketc0lors['candle']['down'], alpha)
        hc = self._mpf_to_rgba(self.marketc0lors.get('hollow', (1.0, 1.0, 1.0, 0.0)))

        colors = self._updownhollow_colors(uc, dc, hc, opens, closes)
        edgecolor = self._updown_colors(self.marketc0lors['edge']['up'], self.marketc0lors['edge']['down'], opens,
                                        closes)
        wickcolor = self._updown_colors(self.marketc0lors['wick']['up'], self.marketc0lors['wick']['down'], opens,
                                        closes)

        lw = self.config.get('_width_config', {}).get('candle_linewidth', 0.8)

        rangeCollection = mcol.LineCollection(rangeSegments, colors=wickcolor, linewidths=lw)
        barCollection = mcol.PolyCollection(barVerts, facecolors=colors, edgecolors=edgecolor, linewidths=lw)

        return [rangeCollection, barCollection]
