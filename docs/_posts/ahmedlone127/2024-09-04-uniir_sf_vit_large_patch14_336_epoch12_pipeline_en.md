---
layout: model
title: English uniir_sf_vit_large_patch14_336_epoch12_pipeline pipeline CLIPForZeroShotClassification from lsr42
author: John Snow Labs
name: uniir_sf_vit_large_patch14_336_epoch12_pipeline
date: 2024-09-04
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

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`uniir_sf_vit_large_patch14_336_epoch12_pipeline` is a English model originally trained by lsr42.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/uniir_sf_vit_large_patch14_336_epoch12_pipeline_en_5.5.0_3.0_1725456809721.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/uniir_sf_vit_large_patch14_336_epoch12_pipeline_en_5.5.0_3.0_1725456809721.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("uniir_sf_vit_large_patch14_336_epoch12_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("uniir_sf_vit_large_patch14_336_epoch12_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|uniir_sf_vit_large_patch14_336_epoch12_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/lsr42/uniir-sf-vit-large-patch14-336-epoch12

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification