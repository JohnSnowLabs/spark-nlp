---
layout: model
title: English syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline pipeline ViTForImageClassification from g30rv17ys
author: John Snow Labs
name: syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline
date: 2025-01-30
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline` is a English model originally trained by g30rv17ys.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline_en_5.5.1_3.0_1738243078962.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline_en_5.5.1_3.0_1738243078962.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|syn_oct_vit_large_4epochs_30c_v2_rundi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.1 GB|

## References

https://huggingface.co/g30rv17ys/syn-oct-ViT-Large-4Epochs-30c-v2-run

## Included Models

- ImageAssembler
- ViTForImageClassification