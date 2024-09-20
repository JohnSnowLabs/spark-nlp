---
layout: model
title: English hub_repoop_1706132005_pipeline pipeline MPNetForSequenceClassification from Kevinger
author: John Snow Labs
name: hub_repoop_1706132005_pipeline
date: 2024-09-02
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

Pretrained MPNetForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hub_repoop_1706132005_pipeline` is a English model originally trained by Kevinger.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hub_repoop_1706132005_pipeline_en_5.5.0_3.0_1725255951856.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hub_repoop_1706132005_pipeline_en_5.5.0_3.0_1725255951856.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hub_repoop_1706132005_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hub_repoop_1706132005_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hub_repoop_1706132005_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.3 MB|

## References

https://huggingface.co/Kevinger/Hub-Repoop-1706132005

## Included Models

- DocumentAssembler
- TokenizerModel
- MPNetForSequenceClassification