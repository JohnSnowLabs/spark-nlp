---
layout: model
title: English google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline pipeline ViTForImageClassification from ZaneHorrible
author: John Snow Labs
name: google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline
date: 2025-02-02
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline` is a English model originally trained by ZaneHorrible.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline_en_5.5.1_3.0_1738478987439.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline_en_5.5.1_3.0_1738478987439.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|google_vit_base_patch16_384_batch_16_epoch_4_classes_24_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.5 MB|

## References

https://huggingface.co/ZaneHorrible/google-vit-base-patch16-384-batch_16_epoch_4_classes_24

## Included Models

- ImageAssembler
- ViTForImageClassification