---
layout: model
title: Modern Greek (1453-) greek_mt5_5ep_384_pipeline pipeline T5Transformer from chaido13
author: John Snow Labs
name: greek_mt5_5ep_384_pipeline
date: 2024-08-09
tags: [el, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: el
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`greek_mt5_5ep_384_pipeline` is a Modern Greek (1453-) model originally trained by chaido13.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/greek_mt5_5ep_384_pipeline_el_5.4.2_3.0_1723220291994.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/greek_mt5_5ep_384_pipeline_el_5.4.2_3.0_1723220291994.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("greek_mt5_5ep_384_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("greek_mt5_5ep_384_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|greek_mt5_5ep_384_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|2.3 GB|

## References

https://huggingface.co/chaido13/greek-mt5-5ep-384

## Included Models

- DocumentAssembler
- T5Transformer