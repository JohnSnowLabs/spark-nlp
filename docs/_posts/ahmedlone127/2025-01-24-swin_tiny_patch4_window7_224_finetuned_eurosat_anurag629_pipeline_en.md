---
layout: model
title: English swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline pipeline SwinForImageClassification from anurag629
author: John Snow Labs
name: swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline
date: 2025-01-24
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

Pretrained SwinForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline` is a English model originally trained by anurag629.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline_en_5.5.1_3.0_1737694362833.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline_en_5.5.1_3.0_1737694362833.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|swin_tiny_patch4_window7_224_finetuned_eurosat_anurag629_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|206.5 MB|

## References

https://huggingface.co/anurag629/swin-tiny-patch4-window7-224-finetuned-eurosat

## Included Models

- ImageAssembler
- SwinForImageClassification