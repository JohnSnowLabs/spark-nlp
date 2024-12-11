---
layout: model
title: English megatron_gpt2_345m_pipeline pipeline GPT2Transformer from robowaifudev
author: John Snow Labs
name: megatron_gpt2_345m_pipeline
date: 2024-12-11
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`megatron_gpt2_345m_pipeline` is a English model originally trained by robowaifudev.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/megatron_gpt2_345m_pipeline_en_5.5.1_3.0_1733906369863.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/megatron_gpt2_345m_pipeline_en_5.5.1_3.0_1733906369863.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("megatron_gpt2_345m_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("megatron_gpt2_345m_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|megatron_gpt2_345m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|970.2 MB|

## References

https://huggingface.co/robowaifudev/megatron-gpt2-345m

## Included Models

- DocumentAssembler
- GPT2Transformer