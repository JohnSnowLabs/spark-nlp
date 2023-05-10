---
layout: model
title: Sentiment Analysis Pipeline for Financial news
author: John Snow Labs
name: classifierdl_bertwiki_finance_sentiment_pipeline
date: 2023-01-12
tags: [en, open_source]
task: Sentiment Analysis
language: en
nav_key: models
edition: Spark NLP 4.3.0
spark_version: 3.0
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bertwiki_finance_sentiment_pipeline_en_4.3.0_3.0_1673543221872.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_bertwiki_finance_sentiment_pipeline_en_4.3.0_3.0_1673543221872.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("classifierdl_bertwiki_finance_sentiment_pipeline", "en")

result = pipeline.annotate("""I love johnsnowlabs!  """)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bertwiki_finance_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|432.9 MB|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel