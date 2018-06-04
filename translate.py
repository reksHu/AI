import urllib.request
import urllib.parse
import json

# content=input("=====请输入您要翻译的内容：=====\n")
content ="天府三街大源国际中心B1栋负一楼中庭花园"

url='http://fanyi.baidu.com/v2transapi'
data={}
data['from']='zh'
data['to']='en'
data['transtype']='translang'
data['simple_means_flag']='3'
data['query']=content
data=urllib.parse.urlencode(data).encode('utf-8')
response=urllib.request.urlopen(url,data)
html=response.read().decode('utf-8')
target=json.loads(html)
print("翻译结果为：%s"%(target['trans_result']['data'][0]['dst']))