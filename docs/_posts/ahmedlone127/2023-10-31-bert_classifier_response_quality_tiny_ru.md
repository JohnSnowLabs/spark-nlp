---
layout: model
title: Russian BertForSequenceClassification Tiny Cased model (from tinkoff-ai)
author: John Snow Labs
name: bert_classifier_response_quality_tiny
date: 2023-10-31
tags: [ru, open_source, bert, sequence_classification, classification, onnx]
task: Text Classification
language: ru
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `response-quality-classifier-tiny` is a Russian model originally trained by `tinkoff-ai`.

## Predicted Entities

`relevance`, `specificity`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_response_quality_tiny_ru_5.1.4_3.4_1698794690965.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_response_quality_tiny_ru_5.1.4_3.4_1698794690965.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_response_quality_tiny","ru") \
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

val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_response_quality_tiny","ru")
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("ru.classify.bert.tiny").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_response_quality_tiny|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ru|
|Size:|109.9 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

References

- https://huggingface.co/tinkoff-ai/response-quality-classifier-tiny
- https://github.com/egoriyaa