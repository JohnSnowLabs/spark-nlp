---
layout: model
title: Sentiment Analysis Pipeline for Financial news
author: John Snow Labs
name: classifierdl_bertwiki_finance_sentiment_pipeline
date: 2021-11-11
tags: [finance, en, bertwiki, classification, sentiment, open_source]
task: Sentiment Analysis
language: en
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline identifies the sentiments (positive, negative or neutral) in financial news.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bertwiki_finance_sentiment_pipeline_en_3.3.0_2.4_1636617651675.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("classifierdl_bertwiki_finance_sentiment_pipeline", lang = "en")

result1 = pipeline.annotate("As interest rates have increased, housing rents have also increased.")
result2 = pipeline.annotate("Unemployment rates have skyrocketed this month.")
result3 = pipeline.annotate("Tax rates on durable consumer goods were reduced.")
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_bertwiki_finance_sentiment_pipeline", lang = "en")

val result1 = pipeline.fullAnnotate("As interest rates have increased, housing rents have also increased.")(0)
val result2 = pipeline.fullAnnotate("Unemployment rates have skyrocketed this month.")(0)
val result3 = pipeline.fullAnnotate("Tax rates on durable consumer goods were reduced.")(0)
```
</div>

## Results

```bash
['neutral']
['negative']
['positive']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bertwiki_finance_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel