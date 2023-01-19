---
layout: model
title: Chinese BertForSequenceClassification Cased model (from kyleinincubated)
author: John Snow Labs
name: bert_classifier_autonlp_abbb_622117836
date: 2022-09-06
tags: [zh, open_source, bert, sequence_classification, classification]
task: Text Classification
language: zh
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autonlp-abbb-622117836` is a Chinese model originally trained by `kyleinincubated`.

## Predicted Entities

`传媒`, `计算机应用`, `化学制药`, `电力`, `汽车租赁`, `拍卖`, `电影`, `生态保护`, `水泥`, `园区开发`, `广播电视`, `港口`, `典当`, `电信`, `种植业`, `畜牧业辅助性活动`, `房地产租赁`, `保障性住房开发`, `燃气`, `小额贷款公司服务`, `保险代理`, `资产管理`, `包装印刷`, `专业公共卫生`, `监护设备`, `文化艺术产业`, `物流`, `保险`, `燃气设备`, `银行`, `商业养老金`, `融资性担保`, `水务`, `中药材种植`, `农产品初加工`, `铁路运输`, `新能源发电`, `中药生产`, `物流配送`, `商业健康保险`, `农产品加工`, `房屋建设`, `医疗人工智能`, `物联网`, `高等教育`, `环保工程`, `珠宝首饰`, `学前教育`, `高速公路服务区`, `娱乐`, `煤炭`, `住宿`, `商品住房开发`, `疾病预防`, `高速公路建设`, `集成电路`, `医疗服务`, `职业技能培训`, `医疗器械`, `贸易`, `燃气供应`, `葡萄酒`, `航空货运`, `证券`, `机场`, `污水处理`, `临床检验`, `中药`, `农村资金互助社服务`, `人身保险`, `锂`, `交通运输`, `水电`, `公用事业`, `纺织服装制造`, `网络安全监测`, `畜禽粪污处理`, `乘用车`, `畜禽养殖`, `航空运输`, `水利工程`, `铁路建设`, `物业管理`, `种子生产`, `保险经纪`, `塑料`, `广播`, `房地产`, `通信设备`, `文化`, `集成电路设计`, `再保险`, `工业园区开发`, `外贸`, `水产品`, `互联网安全服务`, `橡胶`, `互联网服务`, `证券交易`, `贷款公司`, `建筑材料`, `物业服务`, `兽药产品`, `健康体检`, `教育`, `钢铁`, `民宿`, `家具`, `基金`, `基层医疗卫生`, `餐饮`, `普通高中教育`, `建筑业`, `房屋建筑`, `卫星`, `疫苗`, `图书馆`, `航运`, `林业`, `特殊教育`, `体育场馆建筑`, `铝`, `渔业`, `农业`, `公交`, `信托`, `信息技术`, `中等职业学校教育`, `终端设备`, `森林防火`, `证券期货监管服务`, `小额贷款公司`, `互联网平台`, `供水`, `博物馆`, `通信`, `外卖`, `期货`, `公募基金`, `电源设备`, `铁路货物运输`, `房地产中介`, `电视`, `水产养殖`, `旅游综合`, `有色金属`, `传感器设计`, `建筑设计`, `内贸`, `技能培训`, `科技园区开发`, `物联网技术`, `饲料`, `体育`, `医院`, `高速公路`, `超市`, `运行维护服务`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_autonlp_abbb_622117836_zh_4.1.0_3.0_1662502337611.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_autonlp_abbb_622117836_zh_4.1.0_3.0_1662502337611.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autonlp_abbb_622117836","zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autonlp_abbb_622117836","zh") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_autonlp_abbb_622117836|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|zh|
|Size:|384.5 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/kyleinincubated/autonlp-abbb-622117836