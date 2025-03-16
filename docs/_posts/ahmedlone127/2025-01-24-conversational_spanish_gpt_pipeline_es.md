---
layout: model
title: Castilian, Spanish conversational_spanish_gpt_pipeline pipeline GPT2Transformer from ostorc
author: John Snow Labs
name: conversational_spanish_gpt_pipeline
date: 2025-01-24
tags: [es, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`conversational_spanish_gpt_pipeline` is a Castilian, Spanish model originally trained by ostorc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/conversational_spanish_gpt_pipeline_es_5.5.1_3.0_1737716551063.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/conversational_spanish_gpt_pipeline_es_5.5.1_3.0_1737716551063.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("conversational_spanish_gpt_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("conversational_spanish_gpt_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|conversational_spanish_gpt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|466.8 MB|

## References

https://huggingface.co/ostorc/Conversational_Spanish_GPT

## Included Models

- DocumentAssembler
- GPT2Transformer