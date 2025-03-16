---
layout: model
title: Chinese chuvash_construction_vehicle_identification_pipeline pipeline ViTForImageClassification from Bazaar
author: John Snow Labs
name: chuvash_construction_vehicle_identification_pipeline
date: 2025-01-27
tags: [zh, open_source, pipeline, onnx]
task: Image Classification
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`chuvash_construction_vehicle_identification_pipeline` is a Chinese model originally trained by Bazaar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chuvash_construction_vehicle_identification_pipeline_zh_5.5.1_3.0_1737974992800.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chuvash_construction_vehicle_identification_pipeline_zh_5.5.1_3.0_1737974992800.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("chuvash_construction_vehicle_identification_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("chuvash_construction_vehicle_identification_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chuvash_construction_vehicle_identification_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|321.3 MB|

## References

https://huggingface.co/Bazaar/cv_construction_vehicle_identification

## Included Models

- ImageAssembler
- ViTForImageClassification