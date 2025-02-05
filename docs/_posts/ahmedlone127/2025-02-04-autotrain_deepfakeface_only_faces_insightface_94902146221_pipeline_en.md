---
layout: model
title: English autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline pipeline SwinForImageClassification from romitbarua
author: John Snow Labs
name: autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline
date: 2025-02-04
tags: [en, open_source, pipeline, onnx]
task: Image Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained SwinForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline` is a English model originally trained by romitbarua.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline_en_5.5.1_3.0_1738709858995.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline_en_5.5.1_3.0_1738709858995.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_deepfakeface_only_faces_insightface_94902146221_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|206.3 MB|

## References

https://huggingface.co/romitbarua/autotrain-deepfakeface_only_faces_insightface-94902146221

## Included Models

- ImageAssembler
- SwinForImageClassification