---
layout: model
title: Twi albert_news_classification_pipeline pipeline BertForSequenceClassification from clhuang
author: John Snow Labs
name: albert_news_classification_pipeline
date: 2024-09-24
tags: [tw, open_source, pipeline, onnx]
task: Text Classification
language: tw
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_news_classification_pipeline` is a Twi model originally trained by clhuang.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_news_classification_pipeline_tw_5.5.0_3.0_1727213609028.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_news_classification_pipeline_tw_5.5.0_3.0_1727213609028.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_news_classification_pipeline", lang = "tw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_news_classification_pipeline", lang = "tw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_news_classification_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tw|
|Size:|39.8 MB|

## References

https://huggingface.co/clhuang/albert-news-classification

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification