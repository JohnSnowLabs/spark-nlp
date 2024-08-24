---
layout: model
title: Ukrainian ukr_t5_small_pipeline pipeline T5Transformer from d0p3
author: John Snow Labs
name: ukr_t5_small_pipeline
date: 2024-08-24
tags: [uk, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: uk
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ukr_t5_small_pipeline` is a Ukrainian model originally trained by d0p3.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ukr_t5_small_pipeline_uk_5.4.2_3.0_1724459950924.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ukr_t5_small_pipeline_uk_5.4.2_3.0_1724459950924.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ukr_t5_small_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ukr_t5_small_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ukr_t5_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|172.8 MB|

## References

https://huggingface.co/d0p3/ukr-t5-small

## Included Models

- DocumentAssembler
- T5Transformer