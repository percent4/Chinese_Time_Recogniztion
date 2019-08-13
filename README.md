# Chinese_Time_Recogniztion
利用深度学习模型，在小标注量数据上，进行文本中的时间识别。

### 背景介绍

&emsp;&emsp;在文章[NLP入门（十一）从文本中提取时间](https://www.jianshu.com/p/fb4bc83b7dc1) 中，笔者演示了如何利用分词、词性标注的方法从文本中获取时间。当时的想法比较简单快捷，只是利用了词性标注这个功能而已，因此，在某些地方，时间的识别效果并不太好。比如以下的两个例子：

原文1: 
> 苏北大量农村住房建于上世纪80年代之前。去年9月，江苏省决定全面改善苏北农民住房条件，计划3年内改善30万户，作为决胜全面建成小康社会补短板的重要举措。

用笔者之前的代码，提取的时间结果为：

> 提取时间： ['去年9月']

但实际上，我们提取的时间应该是：

> 上世纪80年代之前， 去年9月，3年内

原文2:

> 南宋绍兴十年，金分兵两路向陕西和河南大举进攻，在很快夺回了河南、陕西之后，又率大军向淮南大举进攻。

用笔者之前的代码，提取的时间结果为：

> 提取时间： ['南宋']

但实际上，我们提取的时间应该是：

> 南宋绍兴十年

&emsp;&emsp;因此，利用简单的词性标注功能来提取文本中的时间会存在漏提、错提的情况，鉴于此，笔者想到能否用深度学习模型来实现文本中的时间提取呢？
&emsp;&emsp;该功能类似于命名实体识别（NER）功能，只不过NER是识别文本中的人名、地名、组织机构名，而我们这次需要识别文本中的时间。但是，它们背后的算法原理都是一样的，即采用序列标注模型来解决。

### 项目

&emsp;&emsp;在文章[NLP（十四）自制序列标注平台](https://www.jianshu.com/p/a32bdea77f3e)中，笔者提出了一种自制的序列标注平台，利用该标注平台，笔者从新闻网站中标注了大约2000份语料，标注出文本中的时间，其中75%作为训练集（time.train文件），10%作为验证集（time.dev文件），15%作为测试集（time.test文件）。
&emsp;&emsp;虽然我们现在已经有了深度学习框架方便我们来训练模型，比如TensorFlow, Keras, PyTorch等，但目前已有某大神开源了一个序列标注和文本分类的模块，名称为kashgari-tf，它能够方便快速地用几行命令就可以训练一个序列标注或文本分类的模型，容易上手，而且集中了多种模型（BiGRU，CNN， BiLSTM，CRF）以及多种预训练模型（BERT，ERNIE，wwm-ext），对于用户来说算是十分友好了。该模块的参考网址为：[https://kashgari.bmio.net/](https://kashgari.bmio.net/) 。
&emsp;&emsp;笔者自己花了几天的时间来标注数据，目前已累计标注2000+数据 ，后续将放到Github供大家参考。我们训练的数据，比如time.train的前几行如下：（每一行中间用空格隔开）

```
1 B-TIME
6 I-TIME
0 I-TIME
9 I-TIME
年 I-TIME
， O
日 O
本 O
萨 O
摩 O
藩 O
入 O
侵 O
琉 O
球 O
国 O
， O
并 O
在 O
一 O
个 O
时 O
期 O
内 O
控 O
制 O
琉 O
球 O
国 O
...
```

&emsp;&emsp;接着是模型这块，我们采用经典的BERT+Bi-LSTM+CRF模型，训练1个epoch，batch_size为16，代码如下：

```python
# -*- coding: utf-8 -*-
# time: 2019-08-09 16:47
# place: Zhichunlu Beijing

import kashgari
from kashgari.corpus import DataReader
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

train_x, train_y = DataReader().read_conll_format_file('./data/time.train')
valid_x, valid_y = DataReader().read_conll_format_file('./data/time.dev')
test_x, test_y = DataReader().read_conll_format_file('./data/time.test')

bert_embedding = BERTEmbedding('chinese_L-12_H-768_A-12',
                               task=kashgari.LABELING,
                               sequence_length=128)

model = BiLSTM_CRF_Model(bert_embedding)
model.fit(train_x, train_y, valid_x, valid_y, batch_size=16, epochs=1)

model.save('time_ner.h5')

model.evaluate(test_x, test_y)
```

模型训练完后，得到的效果如下：

|数据集|accuracy|loss|
|---|---|---|
|训练集|0.9814|6.7295|
|验证集|0.6868|150.8513|

在测试集上的结果如下：

|数据集|precision|recall|f1|
|---|---|---|---|
|测试集|0.8547|0.8934|0.8736|

&emsp;&emsp;由于是小标注量，因此我们选择了用BERT预训练模型。如果不采用BERT预训练模型，在同样的数据集上，即使训练100个epoch，虽然在训练集上的准确率超过95%，但是在测试集上却只有大约50%的准确率，效果不行，因此，需要采用预训练模型。

### 测试效果

&emsp;&emsp;在训练完模型后，会在当前目录下生成time_ner.h5模型文件，接着我们需要该模型文件来对新的文件进行预测，提取出文本中的时间。模型预测的代码如下：

```python
# Load saved model
import kashgari

loaded_model = kashgari.utils.load_model('time_ner.h5')

while True:
    text = input('sentence: ')
    t = loaded_model.predict([[char for char in text]])
    print(t)
```

&emsp;&emsp;接着我们在几条新的数据上进行预测，看看该模型的表现效果：

> "原文": "继香港市民10日到“乱港头目”黎智英住所外抗议后，13日，“祸港四人帮”中的另一人李柱铭位于半山的住所外，也有香港市民自发组织前来抗议。",
  "预测时间": [
    "10日",
    "13日"
  ]

> "原文": "绿地控股2018年年度年报显示，截至2018年12月31日，万科金域中央项目的经营状态为“住宅、办公、商业”，项目用地面积18.90万平方米，规划计容建筑面积79.38万平方米，总建筑面积为105.78万平方米，已竣工面积32.90万平方米，总投资额95亿元，报告期实际投资额为10.18亿元。",
  "预测时间": [
    "2018年年度",
    "2018年12月31日"
  ]

> "原文": "经过工作人员两天的反复验证、严密测算，记者昨天从上海中心大厦得到确认：被誉为上海中心大厦“定楼神器”的阻尼器，在8月10日出现自2016年正式启用以来的最大摆幅。",
  "预测时间": [
    "两天",
    "昨天",
    "8月10日",
    "2016年"
  ]

> "原文": "不幸的是，在升任内史的同年九月，狄仁杰就在洛阳私宅离世。",
  "预测时间": [
    "同年九月"
  ]

> "原文": "早上9点25分到达北京火车站，火车站在北京市区哦，地铁很方便到达酒店，我们定了王府井大街的锦江之星，409元一晚，有点小贵。下午去了天坛公园，傍晚去了天安门广场。",
  "预测时间": [
    "早上9点25分",
    "下午",
    "傍晚"
  ],


### 总结

&emsp;&emsp;利用深度学习模型，在小标注量数据上，我们对时间识别取得了不错的效果。后续如果我们想要提高时间识别的准确率，可以再多增加标注数据，目前还只有2000+数据～
&emsp;&emsp;本项目已经开源，Github的地址为：。

&emsp;&emsp;另外，强烈推荐kashgari-tf模块，它能够让你在几分钟内搭建一个序列标注模型，而且方便加载各种预训练模型。

> 注意：不妨了解下笔者的微信公众号： Python爬虫与算法（微信号为：easy_web_scrape）， 欢迎大家关注~
