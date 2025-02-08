---
layout: model
title: Czech gpt2_small_czech_czech_pipeline pipeline GPT2Transformer from spital
author: John Snow Labs
name: gpt2_small_czech_czech_pipeline
date: 2025-02-06
tags: [cs, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: cs
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_small_czech_czech_pipeline` is a Czech model originally trained by spital.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_small_czech_czech_pipeline_cs_5.5.1_3.0_1738856077983.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_small_czech_czech_pipeline_cs_5.5.1_3.0_1738856077983.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_small_czech_czech_pipeline", lang = "cs")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_small_czech_czech_pipeline", lang = "cs")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_small_czech_czech_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|cs|
|Size:|298.1 MB|

## References

https://huggingface.co/spital/gpt2-small-czech-cs

## Included Models

- DocumentAssembler
- GPT2Transformer