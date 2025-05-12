---
layout: model
title: English vit_large_patch16_224_in21k_finetuned_pcam_pipeline pipeline ViTForImageClassification from anurag2op
author: John Snow Labs
name: vit_large_patch16_224_in21k_finetuned_pcam_pipeline
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vit_large_patch16_224_in21k_finetuned_pcam_pipeline` is a English model originally trained by anurag2op.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vit_large_patch16_224_in21k_finetuned_pcam_pipeline_en_5.5.1_3.0_1743887585720.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vit_large_patch16_224_in21k_finetuned_pcam_pipeline_en_5.5.1_3.0_1743887585720.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vit_large_patch16_224_in21k_finetuned_pcam_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vit_large_patch16_224_in21k_finetuned_pcam_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vit_large_patch16_224_in21k_finetuned_pcam_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.1 GB|

## References

https://huggingface.co/anurag2op/vit-large-patch16-224-in21k-finetuned-pcam

## Included Models

- ImageAssembler
- ViTForImageClassification