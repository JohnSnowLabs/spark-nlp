---
layout: model
title: Spanish BertForSequenceClassification Cased model (from gabitoo1234)
author: John Snow Labs
name: bert_sequence_classifier_autotrain_mut_all_text_680820343
date: 2023-11-01
tags: [es, open_source, bert, sequence_classification, ner, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autotrain-mut_all_text-680820343` is a Spanish model originally trained by `gabitoo1234`.

## Predicted Entities

`523.0`, `232.0`, `192.0`, `526.0`, `262.0`, `422.0`, `330.0`, `131.0`, `539.0`, `424.0`, `342.0`, `234.2`, `513.0`, `423.0`, `234.3`, `380.0`, `240.0`, `159.0`, `521.0`, `325.0`, `234.1`, `429.0`, `234.4`, `236.0`, `212.0`, `142.0`, `449.0`, `234.0`, `370.0`, `519.0`, `512.0`, `252.0`, `690.0`, `222.0`, `529.0`, `151.0`, `313.0`, `239.0`, `361.0`, `511.0`, `410.0`, `149.0`, `390.0`, `321.0`, `193.0`, `199.0`, `611.0`, `231.0`, `314.0`, `319.0`, `490.0`, `362.0`, `191.0`, `129.0`, `235.0`, `350.0`, `251.0`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autotrain_mut_all_text_680820343_es_5.1.4_3.4_1698807124594.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autotrain_mut_all_text_680820343_es_5.1.4_3.4_1698807124594.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autotrain_mut_all_text_680820343","es") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autotrain_mut_all_text_680820343","es")
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
|Model Name:|bert_sequence_classifier_autotrain_mut_all_text_680820343|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|412.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/gabitoo1234/autotrain-mut_all_text-680820343