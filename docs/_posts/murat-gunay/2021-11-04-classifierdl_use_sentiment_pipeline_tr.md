---
layout: model
title: Sentiment Analysis Pipeline for Turkish texts
author: John Snow Labs
name: classifierdl_use_sentiment_pipeline
date: 2021-11-04
tags: [turkish, sentiment, tr, open_source]
task: Sentiment Analysis
language: tr
edition: Spark NLP 3.3.1
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline identifies the sentiments (positive or negative) in Turkish texts.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_sentiment_pipeline_tr_3.3.1_2.4_1636020950989.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_use_sentiment_pipeline_tr_3.3.1_2.4_1636020950989.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
|Compatibility:|Spark NLP 3.3.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|

## Included Models

- DocumentAssembler
- UniversalSentenceEncoder
- ClassifierDLModel