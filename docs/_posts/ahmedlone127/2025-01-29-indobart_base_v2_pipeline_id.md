---
layout: model
title: Indonesian indobart_base_v2_pipeline pipeline BartTransformer from gaduhhartawan
author: John Snow Labs
name: indobart_base_v2_pipeline
date: 2025-01-29
tags: [id, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indobart_base_v2_pipeline` is a Indonesian model originally trained by gaduhhartawan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indobart_base_v2_pipeline_id_5.5.1_3.0_1738164402653.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indobart_base_v2_pipeline_id_5.5.1_3.0_1738164402653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indobart_base_v2_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indobart_base_v2_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indobart_base_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|1.9 GB|

## References

https://huggingface.co/gaduhhartawan/indobart-base-v2

## Included Models

- DocumentAssembler
- BartTransformer