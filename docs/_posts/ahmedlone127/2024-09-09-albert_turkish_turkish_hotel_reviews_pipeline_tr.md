---
layout: model
title: Turkish albert_turkish_turkish_hotel_reviews_pipeline pipeline AlbertForSequenceClassification from anilguven
author: John Snow Labs
name: albert_turkish_turkish_hotel_reviews_pipeline
date: 2024-09-09
tags: [tr, open_source, pipeline, onnx]
task: Text Classification
language: tr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_turkish_turkish_hotel_reviews_pipeline` is a Turkish model originally trained by anilguven.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_turkish_turkish_hotel_reviews_pipeline_tr_5.5.0_3.0_1725854608578.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_turkish_turkish_hotel_reviews_pipeline_tr_5.5.0_3.0_1725854608578.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_turkish_turkish_hotel_reviews_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_turkish_turkish_hotel_reviews_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_turkish_turkish_hotel_reviews_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|45.1 MB|

## References

https://huggingface.co/anilguven/albert_tr_turkish_hotel_reviews

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification