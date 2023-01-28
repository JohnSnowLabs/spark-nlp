---
layout: model
title: Sentiment Analysis Pipeline for Turkish texts
author: John Snow Labs
name: classifierdl_use_sentiment_pipeline
date: 2023-01-12
tags: [turkish, sentiment, tr, open_source]
task: Sentiment Analysis
language: tr
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline identifies the sentiments (positive or negative) in Turkish texts.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_sentiment_pipeline_tr_4.3.0_3.0_1673543895690.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_use_sentiment_pipeline_tr_4.3.0_3.0_1673543895690.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("classifierdl_use_sentiment_pipeline", lang = "tr")

result1 = pipeline.annotate("Bu sıralar kafam çok karışık.")
result2 = pipeline.annotate("Sınavımı geçtiğimi öğrenince derin bir nefes aldım.")
```
```scala

val pipeline = new PretrainedPipeline("classifierdl_bert_sentiment_pipeline", lang = "de")

val result1 = pipeline.fullAnnotate("Bu sıralar kafam çok karışık.")(0)
val result2 = pipeline.fullAnnotate("Sınavımı geçtiğimi öğrenince derin bir nefes aldım.")(0)
```
</div>

## Results

```bash

['NEGATIVE']
['POSITIVE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|541.0 MB|

## Included Models

- DocumentAssembler
- UniversalSentenceEncoder
- ClassifierDLModel