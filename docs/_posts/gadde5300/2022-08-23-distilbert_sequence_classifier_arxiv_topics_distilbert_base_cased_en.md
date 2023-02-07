---
layout: model
title: English DistilBertForSequenceClassification Base Cased model (from Wi)
author: John Snow Labs
name: distilbert_sequence_classifier_arxiv_topics_distilbert_base_cased
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

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `arxiv-topics-distilbert-base-cased` is a English model originally trained by `Wi`.

## Predicted Entities

`Other`, `Statistics`, `Astrophysics`, `Quantum Physics`, `Nonlinear Sciences`, `Electrical Engineering and Systems Science`, `High Energy Physics - Lattice`, `Quantitative Biology`, `High Energy Physics - Theory`, `Nuclear Theory`, `High Energy Physics - Experiment`, `Condensed Matter`, `Nuclear Experiment`, `High Energy Physics - Phenomenology`, `Mathematics`, `Physics`, `Quantitative Finance`, `Mathematical Physics`, `Economics`, `General Relativity and Quantum Cosmology`, `Computer Science`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_arxiv_topics_distilbert_base_cased_en_4.1.0_3.0_1661277335053.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_arxiv_topics_distilbert_base_cased_en_4.1.0_3.0_1661277335053.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_arxiv_topics_distilbert_base_cased","en") \
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

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_arxiv_topics_distilbert_base_cased","en") 
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
|Model Name:|distilbert_sequence_classifier_arxiv_topics_distilbert_base_cased|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|246.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Wi/arxiv-topics-distilbert-base-cased