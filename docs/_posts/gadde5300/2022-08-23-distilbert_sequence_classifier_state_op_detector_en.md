---
layout: model
title: English DistilBertForSequenceClassification Cased model (from lingwave-admin)
author: John Snow Labs
name: distilbert_sequence_classifier_state_op_detector
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

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `state-op-detector` is a English model originally trained by `lingwave-admin`.

## Predicted Entities

`State Operator`, `Normal User`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_state_op_detector_en_4.1.0_3.0_1661277955319.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_state_op_detector","en") \
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

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_state_op_detector","en") 
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
|Model Name:|distilbert_sequence_classifier_state_op_detector|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|249.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/lingwave-admin/state-op-detector
- https://www.nbcnews.com/news/latino/russia-disinformation-ukraine-spreading-spanish-speaking-media-rcna22843
- https://github.com/curt-tigges/state-social-operator-detection
- https://www.brookings.edu/techstream/china-and-russia-are-joining-forces-to-spread-disinformation/
- https://journals.sagepub.com/doi/10.1177/19401612221082052
- https://www.brennancenter.org/our-work/analysis-opinion/new-evidence-shows-how-russias-election-interference-has-gotten-more
- https://www.lawfareblog.com/understanding-pro-china-propaganda-and-disinformation-tool-set-xinjiang
- https://www.bbc.com/news/56364952
- https://www.forbes.com/sites/petersuciu/2022/03/10/russian-sock-puppets-spreading-misinformation-on-social-media-about-ukraine/