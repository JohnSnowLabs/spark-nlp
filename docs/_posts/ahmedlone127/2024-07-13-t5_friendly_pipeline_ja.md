---
layout: model
title: Japanese t5_friendly_pipeline pipeline T5Transformer from astremo
author: John Snow Labs
name: t5_friendly_pipeline
date: 2024-07-13
tags: [ja, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ja
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_friendly_pipeline` is a Japanese model originally trained by astremo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_friendly_pipeline_ja_5.4.1_3.0_1720888271327.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_friendly_pipeline_ja_5.4.1_3.0_1720888271327.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_friendly_pipeline", lang = "ja")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_friendly_pipeline", lang = "ja")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_friendly_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|1.0 GB|

## References

https://huggingface.co/astremo/friendly_JA

## Included Models

- DocumentAssembler
- T5Transformer