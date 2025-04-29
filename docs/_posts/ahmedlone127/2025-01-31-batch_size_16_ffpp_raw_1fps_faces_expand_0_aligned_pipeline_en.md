---
layout: model
title: English batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline pipeline SwinForImageClassification from hchcsuim
author: John Snow Labs
name: batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline
date: 2025-01-31
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

Pretrained SwinForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline` is a English model originally trained by hchcsuim.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline_en_5.5.1_3.0_1738342199692.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline_en_5.5.1_3.0_1738342199692.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|batch_size_16_ffpp_raw_1fps_faces_expand_0_aligned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|206.3 MB|

## References

https://huggingface.co/hchcsuim/batch-size-16_FFPP-Raw_1FPS_faces-expand-0-aligned

## Included Models

- ImageAssembler
- SwinForImageClassification