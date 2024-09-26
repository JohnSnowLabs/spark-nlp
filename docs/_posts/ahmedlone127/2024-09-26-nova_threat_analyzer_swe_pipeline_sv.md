---
layout: model
title: Swedish nova_threat_analyzer_swe_pipeline pipeline BertForSequenceClassification from Arro94
author: John Snow Labs
name: nova_threat_analyzer_swe_pipeline
date: 2024-09-26
tags: [sv, open_source, pipeline, onnx]
task: Text Classification
language: sv
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nova_threat_analyzer_swe_pipeline` is a Swedish model originally trained by Arro94.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nova_threat_analyzer_swe_pipeline_sv_5.5.0_3.0_1727321013423.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nova_threat_analyzer_swe_pipeline_sv_5.5.0_3.0_1727321013423.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nova_threat_analyzer_swe_pipeline", lang = "sv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nova_threat_analyzer_swe_pipeline", lang = "sv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nova_threat_analyzer_swe_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sv|
|Size:|467.5 MB|

## References

https://huggingface.co/Arro94/nova-threat-analyzer-swe

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification