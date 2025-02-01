---
layout: model
title: Czech czech_gpt2_oscar_pipeline pipeline GPT2Transformer from lchaloupsky
author: John Snow Labs
name: czech_gpt2_oscar_pipeline
date: 2025-01-29
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`czech_gpt2_oscar_pipeline` is a Czech model originally trained by lchaloupsky.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/czech_gpt2_oscar_pipeline_cs_5.5.1_3.0_1738155950014.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/czech_gpt2_oscar_pipeline_cs_5.5.1_3.0_1738155950014.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("czech_gpt2_oscar_pipeline", lang = "cs")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("czech_gpt2_oscar_pipeline", lang = "cs")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|czech_gpt2_oscar_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|cs|
|Size:|467.0 MB|

## References

https://huggingface.co/lchaloupsky/czech-gpt2-oscar

## Included Models

- DocumentAssembler
- GPT2Transformer