---
layout: model
title: Kannada knt5_base_pipeline pipeline T5Transformer from shraajan
author: John Snow Labs
name: knt5_base_pipeline
date: 2025-01-28
tags: [kn, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: kn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`knt5_base_pipeline` is a Kannada model originally trained by shraajan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/knt5_base_pipeline_kn_5.5.1_3.0_1738094619983.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/knt5_base_pipeline_kn_5.5.1_3.0_1738094619983.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("knt5_base_pipeline", lang = "kn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("knt5_base_pipeline", lang = "kn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|knt5_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|kn|
|Size:|475.9 MB|

## References

https://huggingface.co/shraajan/knt5-base

## Included Models

- DocumentAssembler
- T5Transformer