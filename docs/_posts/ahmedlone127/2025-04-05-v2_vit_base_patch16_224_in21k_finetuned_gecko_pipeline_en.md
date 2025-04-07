---
layout: model
title: English v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline pipeline ViTForImageClassification from emaeon
author: John Snow Labs
name: v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline
date: 2025-04-05
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline` is a English model originally trained by emaeon.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline_en_5.5.1_3.0_1743883855317.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline_en_5.5.1_3.0_1743883855317.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|v2_vit_base_patch16_224_in21k_finetuned_gecko_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.5 MB|

## References

https://huggingface.co/emaeon/v2_vit-base-patch16-224-in21k-finetuned-gecko

## Included Models

- ImageAssembler
- ViTForImageClassification