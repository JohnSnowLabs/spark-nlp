---
layout: model
title: Classifier Pipeline to Identify Hate Speech
author: John Snow Labs
name: bert_sequence_classifier_hatexplain_pipeline
date: 2022-02-22
tags: [bert_for_sequence_classification, hate, hate_speech, speech, offensive, en, open_source]
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

This pretrained pipeline is built on [bert_sequence_classifier_hatexplain_en](https://nlp.johnsnowlabs.com/2021/11/06/bert_sequence_classifier_hatexplain_en.html) model which is imported from `HuggingFace`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_hatexplain_pipeline_en_3.4.0_3.0_1645534173574.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
hatespeech_pipeline = PretrainedPipeline("bert_sequence_classifier_hatexplain_pipeline", lang = "en")

hatespeech_pipeline.annotate("I love you very much!")
```
```scala
val hatespeech_pipeline = new PretrainedPipeline("bert_sequence_classifier_hatexplain_pipeline", lang = "en")

val hatespeech_pipeline.annotate("I love you very much!")
```
</div>

## Results

```bash
['normal']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_hatexplain_pipeline|
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