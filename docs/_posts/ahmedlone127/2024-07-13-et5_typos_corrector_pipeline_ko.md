---
layout: model
title: Korean et5_typos_corrector_pipeline pipeline T5Transformer from j5ng
author: John Snow Labs
name: et5_typos_corrector_pipeline
date: 2024-07-13
tags: [ko, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ko
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`et5_typos_corrector_pipeline` is a Korean model originally trained by j5ng.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/et5_typos_corrector_pipeline_ko_5.4.1_3.0_1720900094505.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/et5_typos_corrector_pipeline_ko_5.4.1_3.0_1720900094505.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("et5_typos_corrector_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("et5_typos_corrector_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|et5_typos_corrector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|1.3 GB|

## References

https://huggingface.co/j5ng/et5-typos-corrector

## Included Models

- DocumentAssembler
- T5Transformer