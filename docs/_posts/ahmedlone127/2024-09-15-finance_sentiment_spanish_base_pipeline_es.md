---
layout: model
title: Castilian, Spanish finance_sentiment_spanish_base_pipeline pipeline BertForSequenceClassification from bardsai
author: John Snow Labs
name: finance_sentiment_spanish_base_pipeline
date: 2024-09-15
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finance_sentiment_spanish_base_pipeline` is a Castilian, Spanish model originally trained by bardsai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finance_sentiment_spanish_base_pipeline_es_5.5.0_3.0_1726376556495.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finance_sentiment_spanish_base_pipeline_es_5.5.0_3.0_1726376556495.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finance_sentiment_spanish_base_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finance_sentiment_spanish_base_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finance_sentiment_spanish_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|411.7 MB|

## References

https://huggingface.co/bardsai/finance-sentiment-es-base

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification