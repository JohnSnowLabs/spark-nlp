---
layout: model
title: English bertbek_news_big_cased_sentiment_apps_pipeline pipeline BertForSequenceClassification from elmurod1202
author: John Snow Labs
name: bertbek_news_big_cased_sentiment_apps_pipeline
date: 2025-04-09
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertbek_news_big_cased_sentiment_apps_pipeline` is a English model originally trained by elmurod1202.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertbek_news_big_cased_sentiment_apps_pipeline_en_5.5.1_3.0_1744165633617.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertbek_news_big_cased_sentiment_apps_pipeline_en_5.5.1_3.0_1744165633617.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertbek_news_big_cased_sentiment_apps_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertbek_news_big_cased_sentiment_apps_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertbek_news_big_cased_sentiment_apps_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.7 MB|

## References

https://huggingface.co/elmurod1202/BERTbek-news-big-cased-sentiment-apps

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification