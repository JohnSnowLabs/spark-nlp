---
layout: model
title: English test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline pipeline SwinForImageClassification from SodaXII
author: John Snow Labs
name: test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline
date: 2025-04-07
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

Pretrained SwinForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline` is a English model originally trained by SodaXII.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline_en_5.5.1_3.0_1744029744911.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline_en_5.5.1_3.0_1744029744911.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|test_swin_base_patch4_window7_224_rice_leaf_disease_augmented_v3_fft_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|649.8 MB|

## References

https://huggingface.co/SodaXII/test_swin-base-patch4-window7-224_rice-leaf-disease-augmented-v3_fft

## Included Models

- ImageAssembler
- SwinForImageClassification