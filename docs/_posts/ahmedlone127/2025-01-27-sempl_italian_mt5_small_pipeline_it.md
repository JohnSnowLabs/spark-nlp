---
layout: model
title: Italian sempl_italian_mt5_small_pipeline pipeline T5Transformer from VerbACxSS
author: John Snow Labs
name: sempl_italian_mt5_small_pipeline
date: 2025-01-27
tags: [it, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: it
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sempl_italian_mt5_small_pipeline` is a Italian model originally trained by VerbACxSS.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sempl_italian_mt5_small_pipeline_it_5.5.1_3.0_1738002347047.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sempl_italian_mt5_small_pipeline_it_5.5.1_3.0_1738002347047.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sempl_italian_mt5_small_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sempl_italian_mt5_small_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sempl_italian_mt5_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|1.2 GB|

## References

https://huggingface.co/VerbACxSS/sempl-it-mt5-small

## Included Models

- DocumentAssembler
- T5Transformer