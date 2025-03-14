---
layout: model
title: Vietnamese gpt2news_pipeline pipeline GPT2Transformer from imthanhlv
author: John Snow Labs
name: gpt2news_pipeline
date: 2025-01-24
tags: [vi, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: vi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2news_pipeline` is a Vietnamese model originally trained by imthanhlv.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2news_pipeline_vi_5.5.1_3.0_1737717588775.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2news_pipeline_vi_5.5.1_3.0_1737717588775.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2news_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2news_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2news_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|466.7 MB|

## References

https://huggingface.co/imthanhlv/gpt2news

## Included Models

- DocumentAssembler
- GPT2Transformer