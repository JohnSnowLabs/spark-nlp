---
layout: model
title: Classifier Pipeline to Identify Antisemitic Texts
author: John Snow Labs
name: bert_sequence_classifier_antisemitism_pipeline
date: 2022-02-22
tags: [bert, sequence_classification, antisemitism, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on [bert_sequence_classifier_antisemitism_en](https://nlp.johnsnowlabs.com/2021/11/06/bert_sequence_classifier_antisemitism_en.html) model which is imported from `HuggingFace`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_antisemitism_pipeline_en_3.4.0_3.0_1645530295089.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
antisemitism_pipeline = PretrainedPipeline("bert_sequence_classifier_antisemitism_pipeline", lang = "en")

antisemitism_pipeline.annotate("The Jews have too much power!")
```
```scala
val antisemitism_pipeline = new PretrainedPipeline("bert_sequence_classifier_antisemitism_pipeline", lang = "en")

val antisemitism_pipeline.annotate("The Jews have too much power!")
```
</div>

## Results

```bash
['1']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_antisemitism_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|410.1 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification