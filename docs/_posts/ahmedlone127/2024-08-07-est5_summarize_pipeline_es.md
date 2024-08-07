---
layout: model
title: Castilian, Spanish est5_summarize_pipeline pipeline T5Transformer from JorgeSarry
author: John Snow Labs
name: est5_summarize_pipeline
date: 2024-08-07
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`est5_summarize_pipeline` is a Castilian, Spanish model originally trained by JorgeSarry.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/est5_summarize_pipeline_es_5.4.2_3.0_1723041126717.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/est5_summarize_pipeline_es_5.4.2_3.0_1723041126717.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("est5_summarize_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("est5_summarize_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|est5_summarize_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|998.1 MB|

## References

https://huggingface.co/JorgeSarry/est5-summarize

## Included Models

- DocumentAssembler
- T5Transformer