---
layout: model
title: Chinese chat_dialogpt_small_chinese_pipeline pipeline GPT2Transformer from liam168
author: John Snow Labs
name: chat_dialogpt_small_chinese_pipeline
date: 2025-03-28
tags: [zh, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`chat_dialogpt_small_chinese_pipeline` is a Chinese model originally trained by liam168.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chat_dialogpt_small_chinese_pipeline_zh_5.5.1_3.0_1743144027903.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chat_dialogpt_small_chinese_pipeline_zh_5.5.1_3.0_1743144027903.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("chat_dialogpt_small_chinese_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("chat_dialogpt_small_chinese_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chat_dialogpt_small_chinese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|467.4 MB|

## References

https://huggingface.co/liam168/chat-DialoGPT-small-zh

## Included Models

- DocumentAssembler
- GPT2Transformer