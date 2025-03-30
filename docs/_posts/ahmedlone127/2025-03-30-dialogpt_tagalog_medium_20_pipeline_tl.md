---
layout: model
title: Tagalog dialogpt_tagalog_medium_20_pipeline pipeline GPT2Transformer from gabtan99
author: John Snow Labs
name: dialogpt_tagalog_medium_20_pipeline
date: 2025-03-30
tags: [tl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: tl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dialogpt_tagalog_medium_20_pipeline` is a Tagalog model originally trained by gabtan99.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dialogpt_tagalog_medium_20_pipeline_tl_5.5.1_3.0_1743359990920.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dialogpt_tagalog_medium_20_pipeline_tl_5.5.1_3.0_1743359990920.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dialogpt_tagalog_medium_20_pipeline", lang = "tl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dialogpt_tagalog_medium_20_pipeline", lang = "tl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dialogpt_tagalog_medium_20_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tl|
|Size:|1.3 GB|

## References

https://huggingface.co/gabtan99/dialogpt-tagalog-medium-20

## Included Models

- DocumentAssembler
- GPT2Transformer