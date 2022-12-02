---
layout: model
title: English BertForSequenceClassification Cased model (from yuan1729)
author: John Snow Labs
name: bert_classifier_non_cl
date: 2022-09-19
tags: [bert, sequence_classification, classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Non_CL` is a English model originally trained by `yuan1729`.

## Predicted Entities

`個人資料保護法`, `竊盜犯贓物犯保安處分條例`, `洗錢防制法`, `商業會計法`, `性騷擾防治法`, `臺灣地區與大陸地區人民關係條例`, `廢棄物清理法`, `著作權法`, `藥事法`, `證券交易法`, `稅捐稽徵法`, `道路交通管理處罰條例`, `電信法`, `中華民國九十六年罪犯減刑條例`, `商標法`, `道路交通安全規則`, `道路交通標誌標線號誌設置規則`, `律師法`, `貪污治罪條例`, `家庭暴力防治法`, `公設辯護人條例`, `通訊保障及監察法`, `轉讓毒品加重其刑之數量標準`, `毒品危害防制條例`, `中華民國憲法`, `就業服務法`, `公司法`, `陸海空軍刑法`, `兒童及少年福利與權益保障法`, `戶籍法`, `兒童及少年性剝削防制條例`, `森林法`, `妨害兵役治罪條例`, `管制藥品管理條例`, `組織犯罪防制條例`, `公職人員選舉罷免法`, `懲治走私條例`, `職業安全衛生法`, `性侵害犯罪防治法`, `水土保持法`, `槍砲彈藥刀械管制條例`, `入出國及移民法`, `罰金罰鍰提高標準條例`, `民法`, `電子遊戲場業管理條例`, `銀行法`, `軍事審判法`, `區域計畫法`, `政府採購法`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_non_cl_en_4.1.0_3.0_1663607318134.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_non_cl","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_non_cl","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_non_cl|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|384.0 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/yuan1729/Non_CL