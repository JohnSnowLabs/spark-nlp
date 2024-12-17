---
layout: model
title: English vit_small_patch16_224_winkawaks_pipeline pipeline ViTForImageClassification from WinKawaks
author: John Snow Labs
name: vit_small_patch16_224_winkawaks_pipeline
date: 2024-12-17
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vit_small_patch16_224_winkawaks_pipeline` is a English model originally trained by WinKawaks.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vit_small_patch16_224_winkawaks_pipeline_en_5.5.1_3.0_1734420707934.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vit_small_patch16_224_winkawaks_pipeline_en_5.5.1_3.0_1734420707934.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vit_small_patch16_224_winkawaks_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vit_small_patch16_224_winkawaks_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vit_small_patch16_224_winkawaks_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|82.6 MB|

## References

https://huggingface.co/WinKawaks/vit-small-patch16-224

## Included Models

- ImageAssembler
- ViTForImageClassification