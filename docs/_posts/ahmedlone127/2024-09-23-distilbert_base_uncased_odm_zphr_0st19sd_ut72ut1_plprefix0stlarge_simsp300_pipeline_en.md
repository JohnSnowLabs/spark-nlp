---
layout: model
title: English distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline pipeline DistilBertForSequenceClassification from tom192180
author: John Snow Labs
name: distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline
date: 2024-09-23
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline` is a English model originally trained by tom192180.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline_en_5.5.0_3.0_1727108509148.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline_en_5.5.0_3.0_1727108509148.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_uncased_odm_zphr_0st19sd_ut72ut1_plprefix0stlarge_simsp300_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.7 MB|

## References

https://huggingface.co/tom192180/distilbert-base-uncased_odm_zphr_0st19sd_ut72ut1_PLPrefix0stlarge_simsp300

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification