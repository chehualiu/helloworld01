
# datapath = ''
datapath = 'option_data\\'

ExHQsvr = '134.175.214.53'
ExHQsvrport = 7727

# HQsvr = '119.147.164.60' # 20220527
HQsvr = '39.105.251.234' # '111.230.189.225' # 20230421
HQsvrport = 7709

# 沪深A股
url_code_hs = 'http://71.push2.eastmoney.com/api/qt/clist/get?cb=jQuery1124024569593216308738_1596880118513&pn=1&pz=20&' \
              'po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,' \
              'm:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,' \
              'f62,f128,f136,f115,f152&_=1596880118514'


headers_code = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip,deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Host': 'myfavor1.eastmoney.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/80.0.3987.149 Safari/537.36'
}