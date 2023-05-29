---
layout: model
title: English BertForSequenceClassification Uncased model (from jonaskoenig)
author: John Snow Labs
name: bert_classifier_xtremedistil_l6_h256_uncased_go_emotion
date: 2022-09-20
tags: [bert, sequence_classification, classification, open_source, en]
task: Text Classification
language: en
nav_key: models
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xtremedistil-l6-h256-uncased-go-emotion` is a English model originally trained by `jonaskoenig`.

## Predicted Entities

`admiration`, `disappointment`, `disgust`, `surprise`, `optimism`, `remorse`, `disapproval`, `anger`, `sadness`, `amusement`, `excitement`, `joy`, `embarrassment`, `love`, `confusion`, `fear`, `approval`, `gratitude`, `neutral`, `caring`, `desire`, `nervousness`, `grief`, `realization`, `curiosity`, `annoyance`, `pride`, `relief`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_xtremedistil_l6_h256_uncased_go_emotion_en_4.2.0_3.0_1663668873480.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_xtremedistil_l6_h256_uncased_go_emotion_en_4.2.0_3.0_1663668873480.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_xtremedistil_l6_h256_uncased_go_emotion","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_xtremedistil_l6_h256_uncased_go_emotion","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.bert.go_emotions.xtremedistiled_uncased").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_xtremedistil_l6_h256_uncased_go_emotion|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|47.6 MB|
|Case sensitive:|false|
|Max sentence length:|256|

## References

- https://huggingface.co/jonaskoenig/xtremedistil-l6-h256-uncased-go-emotion