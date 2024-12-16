---
layout: model
title: Thai wangchan_sentiment_thai_text_model_pipeline pipeline CamemBertForSequenceClassification from phoner45
author: John Snow Labs
name: wangchan_sentiment_thai_text_model_pipeline
date: 2024-12-16
tags: [th, open_source, pipeline, onnx]
task: Text Classification
language: th
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wangchan_sentiment_thai_text_model_pipeline` is a Thai model originally trained by phoner45.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wangchan_sentiment_thai_text_model_pipeline_th_5.5.1_3.0_1734344209019.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wangchan_sentiment_thai_text_model_pipeline_th_5.5.1_3.0_1734344209019.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wangchan_sentiment_thai_text_model_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wangchan_sentiment_thai_text_model_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wangchan_sentiment_thai_text_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|394.4 MB|

## References

https://huggingface.co/phoner45/wangchan-sentiment-thai-text-model

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification