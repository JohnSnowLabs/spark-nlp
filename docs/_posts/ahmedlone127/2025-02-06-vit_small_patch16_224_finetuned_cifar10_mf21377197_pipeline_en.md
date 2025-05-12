---
layout: model
title: English vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline pipeline ViTForImageClassification from MF21377197
author: John Snow Labs
name: vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline
date: 2025-02-06
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline` is a English model originally trained by MF21377197.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline_en_5.5.1_3.0_1738809399203.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline_en_5.5.1_3.0_1738809399203.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vit_small_patch16_224_finetuned_cifar10_mf21377197_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|81.2 MB|

## References

https://huggingface.co/MF21377197/vit-small-patch16-224-finetuned-Cifar10

## Included Models

- ImageAssembler
- ViTForImageClassification