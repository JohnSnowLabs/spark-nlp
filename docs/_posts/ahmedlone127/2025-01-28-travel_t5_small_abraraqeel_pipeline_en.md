---
layout: model
title: English travel_t5_small_abraraqeel_pipeline pipeline T5Transformer from AbrarAqeel
author: John Snow Labs
name: travel_t5_small_abraraqeel_pipeline
date: 2025-01-28
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`travel_t5_small_abraraqeel_pipeline` is a English model originally trained by AbrarAqeel.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/travel_t5_small_abraraqeel_pipeline_en_5.5.1_3.0_1738030677595.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/travel_t5_small_abraraqeel_pipeline_en_5.5.1_3.0_1738030677595.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("travel_t5_small_abraraqeel_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("travel_t5_small_abraraqeel_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|travel_t5_small_abraraqeel_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|275.9 MB|

## References

https://huggingface.co/AbrarAqeel/travel_t5-small

## Included Models

- DocumentAssembler
- T5Transformer