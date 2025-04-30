---
layout: model
title: Castilian, Spanish beto_base_emotion_analysis_pipeline pipeline BertForSequenceClassification from UMUTeam
author: John Snow Labs
name: beto_base_emotion_analysis_pipeline
date: 2025-01-29
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`beto_base_emotion_analysis_pipeline` is a Castilian, Spanish model originally trained by UMUTeam.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/beto_base_emotion_analysis_pipeline_es_5.5.1_3.0_1738150493190.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/beto_base_emotion_analysis_pipeline_es_5.5.1_3.0_1738150493190.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("beto_base_emotion_analysis_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("beto_base_emotion_analysis_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|beto_base_emotion_analysis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|411.8 MB|

## References

https://huggingface.co/UMUTeam/beto-base-emotion-analysis

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification