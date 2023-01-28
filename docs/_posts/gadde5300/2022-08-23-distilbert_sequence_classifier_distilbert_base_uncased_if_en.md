---
layout: model
title: English DistilBertForSequenceClassification Base Uncased model (from Aureliano)
author: John Snow Labs
name: distilbert_sequence_classifier_distilbert_base_uncased_if
date: 2022-08-23
tags: [distilbert, sequence_classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-uncased-if` is a English model originally trained by `Aureliano`.

## Predicted Entities

`charge.v.17`, `kill.v.01`, `put.v.01`, `switch_off.v.01`, `ask.v.01`, `dig.v.01`, `search.v.04`, `repeat.v.01`, `wear.v.02`, `play.v.03`, `ask.v.02`, `wait.v.01`, `smash.v.02`, `clean.v.01`, `drink.v.01`, `inventory.v.01`, `climb.v.01`, `close.v.01`, `set.v.05`, `hit.v.03`, `remove.v.01`, `hit.v.02`, `sit_down.v.01`, `memorize.v.01`, `stand.v.03`, `write.v.07`, `insert.v.01`, `light_up.v.05`, `show.v.01`, `travel.v.01`, `listen.v.01`, `sequence.n.02`, `brandish.v.01`, `take_off.v.06`, `wake_up.v.02`, `connect.v.01`, `say.v.08`, `burn.v.01`, `talk.v.02`, `turn.v.09`, `smell.v.01`, `pull.v.04`, `move.v.02`, `shoot.v.01`, `press.v.01`, `exit.v.01`, `take.v.04`, `examine.v.02`, `read.v.01`, `follow.v.01`, `jump.v.01`, `rub.v.01`, `throw.v.01`, `answer.v.01`, `shake.v.01`, `drive.v.01`, `buy.v.01`, `eat.v.01`, `open.v.01`, `break.v.05`, `note.v.04`, `sleep.v.01`, `drop.v.01`, `blow.v.01`, `fill.v.01`, `choose.v.01`, `enter.v.01`, `pray.v.01`, `skid.v.04`, `lower.v.01`, `lie_down.v.01`, `cut.v.01`, `look.v.01`, `unlock.v.01`, `give.v.03`, `tell.v.03`, `unknown`, `switch_on.v.01`, `consult.v.02`, `raise.v.02`, `insert.v.02`, `pour.v.01`, `touch.v.01`, `push.v.01`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_distilbert_base_uncased_if_en_4.1.0_3.0_1661277779060.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_distilbert_base_uncased_if_en_4.1.0_3.0_1661277779060.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_base_uncased_if","en") \
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

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_base_uncased_if","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_distilbert_base_uncased_if|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|250.0 MB|
|Case sensitive:|false|
|Max sentence length:|128|

## References

- https://huggingface.co/Aureliano/distilbert-base-uncased-if
- https://rasa.com/docs/rasa/components#languagemodelfeaturizer
- https://github.com/aporporato/jericho-corpora