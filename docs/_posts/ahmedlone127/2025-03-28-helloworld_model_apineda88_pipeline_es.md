---
layout: model
title: Castilian, Spanish helloworld_model_apineda88_pipeline pipeline ViTForImageClassification from apineda88
author: John Snow Labs
name: helloworld_model_apineda88_pipeline
date: 2025-03-28
tags: [es, open_source, pipeline, onnx]
task: Image Classification
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`helloworld_model_apineda88_pipeline` is a Castilian, Spanish model originally trained by apineda88.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/helloworld_model_apineda88_pipeline_es_5.5.1_3.0_1743121093752.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/helloworld_model_apineda88_pipeline_es_5.5.1_3.0_1743121093752.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("helloworld_model_apineda88_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("helloworld_model_apineda88_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|helloworld_model_apineda88_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|321.3 MB|

## References

https://huggingface.co/apineda88/helloworld_model

## Included Models

- ImageAssembler
- ViTForImageClassification