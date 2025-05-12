---
layout: model
title: Indonesian simple_plant_detection_pipeline pipeline ViTForImageClassification from novinn
author: John Snow Labs
name: simple_plant_detection_pipeline
date: 2025-03-30
tags: [id, open_source, pipeline, onnx]
task: Image Classification
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`simple_plant_detection_pipeline` is a Indonesian model originally trained by novinn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/simple_plant_detection_pipeline_id_5.5.1_3.0_1743336052628.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/simple_plant_detection_pipeline_id_5.5.1_3.0_1743336052628.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("simple_plant_detection_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("simple_plant_detection_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|simple_plant_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|321.4 MB|

## References

https://huggingface.co/novinn/simple-plant-detection

## Included Models

- ImageAssembler
- ViTForImageClassification