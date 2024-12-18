---
layout: model
title: Hebrew hebrew_gpt2_345m_stage_pipeline pipeline GPT2Transformer from Norod78
author: John Snow Labs
name: hebrew_gpt2_345m_stage_pipeline
date: 2024-12-17
tags: [he, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: he
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hebrew_gpt2_345m_stage_pipeline` is a Hebrew model originally trained by Norod78.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hebrew_gpt2_345m_stage_pipeline_he_5.5.1_3.0_1734394101847.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hebrew_gpt2_345m_stage_pipeline_he_5.5.1_3.0_1734394101847.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hebrew_gpt2_345m_stage_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hebrew_gpt2_345m_stage_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hebrew_gpt2_345m_stage_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|1.5 GB|

## References

https://huggingface.co/Norod78/Hebrew-GPT2-345M-Stage

## Included Models

- DocumentAssembler
- GPT2Transformer