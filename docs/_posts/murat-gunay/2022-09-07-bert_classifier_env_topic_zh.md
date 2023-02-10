---
layout: model
title: Chinese BertForSequenceClassification Cased model (from celtics1863)
author: John Snow Labs
name: bert_classifier_env_topic
date: 2022-09-07
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `env-bert-topic` is a Chinese model originally trained by `celtics1863`.

## Predicted Entities

`邻居噪音`, `空气污染`, `节能减排`, `环保组织`, `噪音`, `自来水`, `环评工程师`, `碳金融`, `穹顶之下（纪录片）`, `环境污染`, `地球一小时`, `净水器`, `野生动物保护`, `雾霾`, `雾霾治理`, `建筑节能`, `低碳`, `化学污染`, `PM 2.5`, `垃圾处理`, `水污染`, `噪音污染`, `沙尘暴`, `植树`, `垃圾分类`, `环境科学`, `环境评估`, `核污染`, `核能`, `污染治理`, `物种多样性`, `自然环境`, `二氧化碳`, `沙漠治理`, `水处理工程`, `水处理`, `风能及风力发电`, `重金属污染`, `温室效应`, `生活`, `垃圾焚烧`, `环保督查`, `环保行业`, `生物多样性`, `净水设备`, `生态环境`, `环境伦理学`, `污水处理`, `工业污染`, `土壤污染`, `垃圾处理器`, `气候变化`, `环境工程`, `太阳能`, `全球变暖`, `碳交易`, `绿色建筑`, `巴黎协定`, `环境保护`, `环境评价`, `垃圾`, `碳排放`, `环保经济`, `污水排放`, `太空垃圾`, `温室气体`, `净水`, `秸秆焚烧`, `室内空气污染`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_env_topic_zh_4.1.0_3.0_1662512783236.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_env_topic_zh_4.1.0_3.0_1662512783236.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_env_topic","zh") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_env_topic","zh") 
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
|Model Name:|bert_classifier_env_topic|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|zh|
|Size:|384.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/celtics1863/env-bert-topic