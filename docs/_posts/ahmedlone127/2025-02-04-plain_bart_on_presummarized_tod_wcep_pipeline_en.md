---
layout: model
title: English plain_bart_on_presummarized_tod_wcep_pipeline pipeline BartTransformer from roofdancer
author: John Snow Labs
name: plain_bart_on_presummarized_tod_wcep_pipeline
date: 2025-02-04
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`plain_bart_on_presummarized_tod_wcep_pipeline` is a English model originally trained by roofdancer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/plain_bart_on_presummarized_tod_wcep_pipeline_en_5.5.1_3.0_1738704861501.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/plain_bart_on_presummarized_tod_wcep_pipeline_en_5.5.1_3.0_1738704861501.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("plain_bart_on_presummarized_tod_wcep_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("plain_bart_on_presummarized_tod_wcep_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|plain_bart_on_presummarized_tod_wcep_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/roofdancer/plain-bart-on-presummarized-tod-wcep

## Included Models

- DocumentAssembler
- BartTransformer