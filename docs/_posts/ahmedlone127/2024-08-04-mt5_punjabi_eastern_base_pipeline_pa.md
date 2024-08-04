---
layout: model
title: Panjabi, Punjabi mt5_punjabi_eastern_base_pipeline pipeline T5Transformer from rukaiyaaaah
author: John Snow Labs
name: mt5_punjabi_eastern_base_pipeline
date: 2024-08-04
tags: [pa, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: pa
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_punjabi_eastern_base_pipeline` is a Panjabi, Punjabi model originally trained by rukaiyaaaah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_punjabi_eastern_base_pipeline_pa_5.4.2_3.0_1722756831340.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_punjabi_eastern_base_pipeline_pa_5.4.2_3.0_1722756831340.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_punjabi_eastern_base_pipeline", lang = "pa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_punjabi_eastern_base_pipeline", lang = "pa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_punjabi_eastern_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|pa|
|Size:|471.7 MB|

## References

https://huggingface.co/rukaiyaaaah/mt5-pa-base

## Included Models

- DocumentAssembler
- T5Transformer