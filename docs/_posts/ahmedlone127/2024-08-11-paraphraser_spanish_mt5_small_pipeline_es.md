---
layout: model
title: Castilian, Spanish paraphraser_spanish_mt5_small_pipeline pipeline T5Transformer from milyiyo
author: John Snow Labs
name: paraphraser_spanish_mt5_small_pipeline
date: 2024-08-11
tags: [es, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: es
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`paraphraser_spanish_mt5_small_pipeline` is a Castilian, Spanish model originally trained by milyiyo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/paraphraser_spanish_mt5_small_pipeline_es_5.4.2_3.0_1723394865147.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/paraphraser_spanish_mt5_small_pipeline_es_5.4.2_3.0_1723394865147.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("paraphraser_spanish_mt5_small_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("paraphraser_spanish_mt5_small_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|paraphraser_spanish_mt5_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|1.2 GB|

## References

https://huggingface.co/milyiyo/paraphraser-spanish-mt5-small

## Included Models

- DocumentAssembler
- T5Transformer