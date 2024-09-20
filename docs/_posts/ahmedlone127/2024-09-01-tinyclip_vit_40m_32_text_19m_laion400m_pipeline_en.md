---
layout: model
title: English tinyclip_vit_40m_32_text_19m_laion400m_pipeline pipeline CLIPForZeroShotClassification from wkcn
author: John Snow Labs
name: tinyclip_vit_40m_32_text_19m_laion400m_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tinyclip_vit_40m_32_text_19m_laion400m_pipeline` is a English model originally trained by wkcn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tinyclip_vit_40m_32_text_19m_laion400m_pipeline_en_5.4.2_3.0_1725159269631.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tinyclip_vit_40m_32_text_19m_laion400m_pipeline_en_5.4.2_3.0_1725159269631.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tinyclip_vit_40m_32_text_19m_laion400m_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tinyclip_vit_40m_32_text_19m_laion400m_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tinyclip_vit_40m_32_text_19m_laion400m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|202.5 MB|

## References

https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification