---
layout: model
title: Indonesian BertForSequenceClassification Base Cased model (from ayameRushia)
author: John Snow Labs
name: bert_sequence_classifier_base_indonesian_1.5g_sentiment_analysis_smsa
date: 2023-10-26
tags: [id, open_source, bert, sequence_classification, ner, onnx]
task: Named Entity Recognition
language: id
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-indonesian-1.5G-sentiment-analysis-smsa` is a Indonesian model originally trained by `ayameRushia`.

## Predicted Entities

`Neutral`, `Positive`, `Negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_base_indonesian_1.5g_sentiment_analysis_smsa_id_5.1.4_3.4_1698312586345.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_base_indonesian_1.5g_sentiment_analysis_smsa_id_5.1.4_3.4_1698312586345.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_base_indonesian_1.5g_sentiment_analysis_smsa","id") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_base_indonesian_1.5g_sentiment_analysis_smsa","id")
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
|Model Name:|bert_sequence_classifier_base_indonesian_1.5g_sentiment_analysis_smsa|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|id|
|Size:|414.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa
- https://paperswithcode.com/sota?task=Text+Classification&dataset=indonlu