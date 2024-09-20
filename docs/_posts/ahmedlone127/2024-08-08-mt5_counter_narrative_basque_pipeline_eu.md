---
layout: model
title: Basque mt5_counter_narrative_basque_pipeline pipeline T5Transformer from HiTZ
author: John Snow Labs
name: mt5_counter_narrative_basque_pipeline
date: 2024-08-08
tags: [eu, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: eu
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_counter_narrative_basque_pipeline` is a Basque model originally trained by HiTZ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_counter_narrative_basque_pipeline_eu_5.4.2_3.0_1723137499322.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_counter_narrative_basque_pipeline_eu_5.4.2_3.0_1723137499322.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_counter_narrative_basque_pipeline", lang = "eu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_counter_narrative_basque_pipeline", lang = "eu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_counter_narrative_basque_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|eu|
|Size:|2.2 GB|

## References

https://huggingface.co/HiTZ/mt5-counter-narrative-eu

## Included Models

- DocumentAssembler
- T5Transformer