---
layout: model
title: English xsum_samsum_threshold0_25_epoch3_pipeline pipeline BartTransformer from PSW
author: John Snow Labs
name: xsum_samsum_threshold0_25_epoch3_pipeline
date: 2025-04-01
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xsum_samsum_threshold0_25_epoch3_pipeline` is a English model originally trained by PSW.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xsum_samsum_threshold0_25_epoch3_pipeline_en_5.5.1_3.0_1743520139868.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xsum_samsum_threshold0_25_epoch3_pipeline_en_5.5.1_3.0_1743520139868.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xsum_samsum_threshold0_25_epoch3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xsum_samsum_threshold0_25_epoch3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xsum_samsum_threshold0_25_epoch3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|810.5 MB|

## References

https://huggingface.co/PSW/xsum_samsum_threshold0.25_epoch3

## Included Models

- DocumentAssembler
- BartTransformer