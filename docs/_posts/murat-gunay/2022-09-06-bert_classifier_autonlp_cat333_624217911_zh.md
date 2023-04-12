---
layout: model
title: Chinese BertForSequenceClassification Cased model (from kyleinincubated)
author: John Snow Labs
name: bert_classifier_autonlp_cat333_624217911
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autonlp-cat333-624217911` is a Chinese model originally trained by `kyleinincubated`.

## Predicted Entities

`渔业`, `采矿业`, `公用事业`, `交通运输`, `农业`, `电子制造`, `休闲服务`, `文化`, `商业贸易`, `畜牧业`, `林业`, `轻工制造`, `教育`, `食品饮料`, `化工制造`, `非银金融`, `房地产`, `传媒`, `通信`, `汽车制造`, `信息技术`, `有色金属`, `互联网服务`, `银行`, `纺织服装制造`, `医药生物`, `钢铁`, `建筑业`, `电气设备`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_autonlp_cat333_624217911_zh_4.1.0_3.0_1662503009543.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_autonlp_cat333_624217911_zh_4.1.0_3.0_1662503009543.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autonlp_cat333_624217911","zh") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autonlp_cat333_624217911","zh") 
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
|Model Name:|bert_classifier_autonlp_cat333_624217911|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|zh|
|Size:|384.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/kyleinincubated/autonlp-cat333-624217911