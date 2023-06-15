---
layout: model
title: Mongolian RobertaForTokenClassification Base Cased model (from onon214)
author: John Snow Labs
name: roberta_token_classifier_base_ner_demo
date: 2023-03-01
tags: [mn, open_source, roberta, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: mn
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-base-ner-demo` is a Mongolian model originally trained by `onon214`.

## Predicted Entities

`MISC`, `LOC`, `PER`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_base_ner_demo_mn_4.3.0_3.0_1677703536380.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_base_ner_demo_mn_4.3.0_3.0_1677703536380.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_base_ner_demo","mn") \
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

val tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_base_ner_demo","mn")
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
|Model Name:|roberta_token_classifier_base_ner_demo|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|mn|
|Size:|466.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/onon214/roberta-base-ner-demo