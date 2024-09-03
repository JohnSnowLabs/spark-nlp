---
layout: model
title: English clip_fashion_attribute_model_try_2_base32_pipeline pipeline CLIPForZeroShotClassification from Geetansh13
author: John Snow Labs
name: clip_fashion_attribute_model_try_2_base32_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
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

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clip_fashion_attribute_model_try_2_base32_pipeline` is a English model originally trained by Geetansh13.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clip_fashion_attribute_model_try_2_base32_pipeline_en_5.5.0_3.0_1725339266748.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clip_fashion_attribute_model_try_2_base32_pipeline_en_5.5.0_3.0_1725339266748.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("clip_fashion_attribute_model_try_2_base32_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("clip_fashion_attribute_model_try_2_base32_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clip_fashion_attribute_model_try_2_base32_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|580.7 MB|

## References

https://huggingface.co/Geetansh13/clip-fashion-attribute-model-try-2-base32

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification