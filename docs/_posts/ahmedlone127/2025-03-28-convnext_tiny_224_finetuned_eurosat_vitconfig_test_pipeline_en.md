---
layout: model
title: English convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline pipeline ViTForImageClassification from polejowska
author: John Snow Labs
name: convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline
date: 2025-03-28
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline` is a English model originally trained by polejowska.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline_en_5.5.1_3.0_1743135272757.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline_en_5.5.1_3.0_1743135272757.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|convnext_tiny_224_finetuned_eurosat_vitconfig_test_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|327.5 MB|

## References

https://huggingface.co/polejowska/convnext-tiny-224-finetuned-eurosat-vitconfig-test

## Included Models

- ImageAssembler
- ViTForImageClassification