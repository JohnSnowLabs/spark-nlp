---
layout: model
title: Danish BertForTokenClassification Base Cased model (from alexandrainst)
author: John Snow Labs
name: bert_token_classifier_da_ner_base
date: 2023-03-20
tags: [da, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: da
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `da-ner-base` is a Danish model originally trained by `alexandrainst`.

## Predicted Entities

`LOC`, `ORG`, `PER`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_da_ner_base_da_4.3.1_3.0_1679333362426.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_da_ner_base_da_4.3.1_3.0_1679333362426.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_da_ner_base","da") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_da_ner_base","da")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_da_ner_base|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|da|
|Size:|412.9 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/alexandrainst/da-ner-base
- https://github.com/certainlyio/nordic_bert
- https://danlp-alexandra.readthedocs.io/en/latest/docs/tasks/ner.html#bert
- https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#dane