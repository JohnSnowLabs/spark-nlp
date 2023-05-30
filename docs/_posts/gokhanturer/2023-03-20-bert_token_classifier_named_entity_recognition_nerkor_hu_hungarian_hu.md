---
layout: model
title: Hungarian BertForTokenClassification Cased model (from NYTK)
author: John Snow Labs
name: bert_token_classifier_named_entity_recognition_nerkor_hu_hungarian
date: 2023-03-20
tags: [hu, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: hu
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `named-entity-recognition-nerkor-hubert-hungarian` is a Hungarian model originally trained by `NYTK`.

## Predicted Entities

`LOC`, `ORG`, `PER`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_named_entity_recognition_nerkor_hu_hungarian_hu_4.3.1_3.0_1679333446245.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_named_entity_recognition_nerkor_hu_hungarian_hu_4.3.1_3.0_1679333446245.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_named_entity_recognition_nerkor_hu_hungarian","hu") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_named_entity_recognition_nerkor_hu_hungarian","hu")
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
|Model Name:|bert_token_classifier_named_entity_recognition_nerkor_hu_hungarian|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|hu|
|Size:|413.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/NYTK/named-entity-recognition-nerkor-hubert-hungarian
- https://juniper.nytud.hu/demo/nlp
- https://github.com/nytud/NYTK-NerKor