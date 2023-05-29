---
layout: model
title: Spanish BertForSequenceClassification Cased model (from finiteautomata)
author: John Snow Labs
name: bert_classifier_non_contextualized_hate_speech
date: 2022-09-19
tags: [bert, sequence_classification, classification, open_source, es]
task: Text Classification
language: es
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-non-contextualized-hate-speech-es` is a Spanish model originally trained by `finiteautomata`.

## Predicted Entities

`Hateful`, `Not hateful`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_non_contextualized_hate_speech_es_4.1.0_3.0_1663607985579.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_non_contextualized_hate_speech_es_4.1.0_3.0_1663607985579.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_non_contextualized_hate_speech","es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["Amo Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_non_contextualized_hate_speech","es") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("Amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.classify.hate_non_contextualized.bert.by_finiteautomata").predict("""Amo Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_non_contextualized_hate_speech|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|412.4 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/finiteautomata/bert-non-contextualized-hate-speech-es