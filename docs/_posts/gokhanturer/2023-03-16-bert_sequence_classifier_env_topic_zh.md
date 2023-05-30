---
layout: model
title: Chinese BertForSequenceClassification Cased model (from celtics1863)
author: John Snow Labs
name: bert_sequence_classifier_env_topic
date: 2023-03-16
tags: [zh, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: zh
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `env-bert-topic` is a Chinese model originally trained by `celtics1863`.

## Predicted Entities

`沙尘暴`, `沙漠治理`, `温室气体`, `污水处理`, `地球一小时`, `PM 2.5`, `碳金融`, `环保经济`, `环保行业`, `环评工程师`, `太阳能`, `核污染`, `环境伦理学`, `空气污染`, `重金属污染`, `化学污染`, `自来水`, `水处理`, `环境评估`, `巴黎协定`, `碳排放`, `温室效应`, `雾霾`, `低碳`, `垃圾焚烧`, `污水排放`, `室内空气污染`, `物种多样性`, `垃圾`, `环保组织`, `垃圾分类`, `垃圾处理`, `环境污染`, `净水`, `植树`, `生物多样性`, `环境工程`, `穹顶之下（纪录片）`, `太空垃圾`, `水污染`, `核能`, `气候变化`, `环保督查`, `工业污染`, `水处理工程`, `环境科学`, `环境保护`, `净水器`, `秸秆焚烧`, `自然环境`, `建筑节能`, `碳交易`, `净水设备`, `邻居噪音`, `节能减排`, `风能及风力发电`, `二氧化碳`, `雾霾治理`, `绿色建筑`, `噪音污染`, `噪音`, `污染治理`, `生活`, `全球变暖`, `野生动物保护`, `环境评价`, `生态环境`, `土壤污染`, `垃圾处理器`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_env_topic_zh_4.3.1_3.0_1678985419884.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_env_topic_zh_4.3.1_3.0_1678985419884.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_env_topic","zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_env_topic","zh")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_env_topic|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|384.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/celtics1863/env-bert-topic