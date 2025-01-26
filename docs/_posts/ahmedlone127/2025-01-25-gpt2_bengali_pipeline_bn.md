---
layout: model
title: Bengali gpt2_bengali_pipeline pipeline GPT2Transformer from flax-community
author: John Snow Labs
name: gpt2_bengali_pipeline
date: 2025-01-25
tags: [bn, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: bn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_bengali_pipeline` is a Bengali model originally trained by flax-community.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_bengali_pipeline_bn_5.5.1_3.0_1737766044240.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_bengali_pipeline_bn_5.5.1_3.0_1737766044240.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_bengali_pipeline", lang = "bn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_bengali_pipeline", lang = "bn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_bengali_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|bn|
|Size:|467.2 MB|

## References

https://huggingface.co/flax-community/gpt2-bengali

## Included Models

- DocumentAssembler
- GPT2Transformer