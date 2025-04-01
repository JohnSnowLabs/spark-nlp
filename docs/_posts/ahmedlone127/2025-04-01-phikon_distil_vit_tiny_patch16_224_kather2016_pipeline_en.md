---
layout: model
title: English phikon_distil_vit_tiny_patch16_224_kather2016_pipeline pipeline ViTForImageClassification from 1aurent
author: John Snow Labs
name: phikon_distil_vit_tiny_patch16_224_kather2016_pipeline
date: 2025-04-01
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`phikon_distil_vit_tiny_patch16_224_kather2016_pipeline` is a English model originally trained by 1aurent.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/phikon_distil_vit_tiny_patch16_224_kather2016_pipeline_en_5.5.1_3.0_1743494346760.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/phikon_distil_vit_tiny_patch16_224_kather2016_pipeline_en_5.5.1_3.0_1743494346760.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("phikon_distil_vit_tiny_patch16_224_kather2016_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("phikon_distil_vit_tiny_patch16_224_kather2016_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|phikon_distil_vit_tiny_patch16_224_kather2016_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|20.7 MB|

## References

https://huggingface.co/1aurent/phikon-distil-vit-tiny-patch16-224-kather2016

## Included Models

- ImageAssembler
- ViTForImageClassification