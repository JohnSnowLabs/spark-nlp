---
layout: model
title: Korean ke_t5_small_newslike_pipeline pipeline T5Transformer from KETI-AIR
author: John Snow Labs
name: ke_t5_small_newslike_pipeline
date: 2024-08-06
tags: [ko, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ko
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ke_t5_small_newslike_pipeline` is a Korean model originally trained by KETI-AIR.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ke_t5_small_newslike_pipeline_ko_5.4.2_3.0_1722907769008.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ke_t5_small_newslike_pipeline_ko_5.4.2_3.0_1722907769008.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ke_t5_small_newslike_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ke_t5_small_newslike_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ke_t5_small_newslike_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|273.4 MB|

## References

https://huggingface.co/KETI-AIR/ke-t5-small-newslike

## Included Models

- DocumentAssembler
- T5Transformer