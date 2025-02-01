---
layout: model
title: Norwegian sammendrag_pipeline pipeline BartTransformer from norkart
author: John Snow Labs
name: sammendrag_pipeline
date: 2025-01-31
tags: ["no", open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: "no"
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sammendrag_pipeline` is a Norwegian model originally trained by norkart.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sammendrag_pipeline_no_5.5.1_3.0_1738295665106.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sammendrag_pipeline_no_5.5.1_3.0_1738295665106.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sammendrag_pipeline", lang = "no")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sammendrag_pipeline", lang = "no")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sammendrag_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|no|
|Size:|1.9 GB|

## References

https://huggingface.co/norkart/sammendrag

## Included Models

- DocumentAssembler
- BartTransformer