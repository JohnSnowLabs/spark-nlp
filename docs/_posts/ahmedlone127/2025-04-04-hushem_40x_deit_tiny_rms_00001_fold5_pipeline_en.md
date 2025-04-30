---
layout: model
title: English hushem_40x_deit_tiny_rms_00001_fold5_pipeline pipeline ViTForImageClassification from hkivancoral
author: John Snow Labs
name: hushem_40x_deit_tiny_rms_00001_fold5_pipeline
date: 2025-04-04
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hushem_40x_deit_tiny_rms_00001_fold5_pipeline` is a English model originally trained by hkivancoral.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hushem_40x_deit_tiny_rms_00001_fold5_pipeline_en_5.5.1_3.0_1743793364444.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hushem_40x_deit_tiny_rms_00001_fold5_pipeline_en_5.5.1_3.0_1743793364444.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hushem_40x_deit_tiny_rms_00001_fold5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hushem_40x_deit_tiny_rms_00001_fold5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hushem_40x_deit_tiny_rms_00001_fold5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|20.8 MB|

## References

https://huggingface.co/hkivancoral/hushem_40x_deit_tiny_rms_00001_fold5

## Included Models

- ImageAssembler
- ViTForImageClassification