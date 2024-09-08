---
layout: model
title: Italian lld_valbadia_ita_loresmt_l4_pipeline pipeline MarianTransformer from sfrontull
author: John Snow Labs
name: lld_valbadia_ita_loresmt_l4_pipeline
date: 2024-09-07
tags: [it, open_source, pipeline, onnx]
task: Translation
language: it
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`lld_valbadia_ita_loresmt_l4_pipeline` is a Italian model originally trained by sfrontull.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lld_valbadia_ita_loresmt_l4_pipeline_it_5.5.0_3.0_1725740935736.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/lld_valbadia_ita_loresmt_l4_pipeline_it_5.5.0_3.0_1725740935736.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("lld_valbadia_ita_loresmt_l4_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("lld_valbadia_ita_loresmt_l4_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lld_valbadia_ita_loresmt_l4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|410.9 MB|

## References

https://huggingface.co/sfrontull/lld_valbadia-ita-loresmt-L4

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer