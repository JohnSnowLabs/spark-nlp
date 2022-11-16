---
layout: model
title: German DistilBertForSequenceClassification Base Cased model (from ml6team)
author: John Snow Labs
name: distilbert_sequence_classifier_distilbert_base_german_cased_toxic_comments
date: 2022-08-23
tags: [distilbert, sequence_classification, open_source, de]
task: Text Classification
language: de
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-german-cased-toxic-comments` is a German model originally trained by `ml6team`.

## Predicted Entities

`toxic`, `non_toxic`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_distilbert_base_german_cased_toxic_comments_de_4.1.0_3.0_1661277647839.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_base_german_cased_toxic_comments","de") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["Ich liebe Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_base_german_cased_toxic_comments","de") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("Ich liebe Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_distilbert_base_german_cased_toxic_comments|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|252.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ml6team/distilbert-base-german-cased-toxic-comments
- http://ub-web.de/research/
- https://github.com/uds-lsv/GermEval-2018-Data
- https://arxiv.org/pdf/1701.08118.pdf
- https://github.com/UCSM-DUE/IWG_hatespeech_public
- https://hasocfire.github.io/hasoc/2019/index.html
- https://github.com/germeval2021toxic/SharedTask/tree/main/Data%20Sets