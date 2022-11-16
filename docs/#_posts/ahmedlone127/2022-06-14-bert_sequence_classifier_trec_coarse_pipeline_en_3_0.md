---
layout: model
title: Pipeline to Classify Texts into TREC-6 Classes
author: ahmedlone127
name: bert_sequence_classifier_trec_coarse_pipeline
date: 2022-06-14
tags: [bert_sequence, trec, coarse, bert, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_sequence_classifier_trec_coarse_en](https://nlp.johnsnowlabs.com/2021/11/06/bert_sequence_classifier_trec_coarse_en.html).

The TREC dataset for question classification consists of open-domain, fact-based questions divided into broad semantic categories. You can check the official documentation of the dataset, entities, etc. [here](https://search.r-project.org/CRAN/refmans/textdata/html/dataset_trec.html).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/ahmedlone127/bert_sequence_classifier_trec_coarse_pipeline_en_4.0.0_3.0_1655211903928.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

trec_pipeline = PretrainedPipeline("bert_sequence_classifier_trec_coarse_pipeline", lang = "en")

trec_pipeline.annotate("Germany is the largest country in Europe economically.")
```
```scala

val trec_pipeline = new PretrainedPipeline("bert_sequence_classifier_trec_coarse_pipeline", lang = "en")

trec_pipeline.annotate("Germany is the largest country in Europe economically.")
```
</div>

## Results

```bash

['LOC']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_trec_coarse_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Community|
|Language:|en|
|Size:|406.6 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification