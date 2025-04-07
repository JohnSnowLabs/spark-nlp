---
layout: model
title: Persian xls_r_1b_persian_farsi_cv8_pipeline pipeline Wav2Vec2ForCTC from ghofrani
author: John Snow Labs
name: xls_r_1b_persian_farsi_cv8_pipeline
date: 2025-04-07
tags: [fa, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xls_r_1b_persian_farsi_cv8_pipeline` is a Persian model originally trained by ghofrani.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xls_r_1b_persian_farsi_cv8_pipeline_fa_5.5.1_3.0_1744010818725.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xls_r_1b_persian_farsi_cv8_pipeline_fa_5.5.1_3.0_1744010818725.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xls_r_1b_persian_farsi_cv8_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xls_r_1b_persian_farsi_cv8_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xls_r_1b_persian_farsi_cv8_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|1.2 GB|

## References

https://huggingface.co/ghofrani/xls-r-1b-fa-cv8

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC