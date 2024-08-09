---
layout: model
title: Chinese mt5_translate_yue_chinese_chinese_pipeline pipeline T5Transformer from botisan-ai
author: John Snow Labs
name: mt5_translate_yue_chinese_chinese_pipeline
date: 2024-07-31
tags: [zh, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: zh
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_translate_yue_chinese_chinese_pipeline` is a Chinese model originally trained by botisan-ai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_translate_yue_chinese_chinese_pipeline_zh_5.4.2_3.0_1722443566333.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_translate_yue_chinese_chinese_pipeline_zh_5.4.2_3.0_1722443566333.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_translate_yue_chinese_chinese_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_translate_yue_chinese_chinese_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_translate_yue_chinese_chinese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|2.2 GB|

## References

https://huggingface.co/botisan-ai/mt5-translate-yue-zh

## Included Models

- DocumentAssembler
- T5Transformer