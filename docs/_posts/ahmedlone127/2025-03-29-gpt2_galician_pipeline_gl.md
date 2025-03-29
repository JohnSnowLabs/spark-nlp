---
layout: model
title: Galician gpt2_galician_pipeline pipeline GPT2Transformer from fpuentes
author: John Snow Labs
name: gpt2_galician_pipeline
date: 2025-03-29
tags: [gl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: gl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_galician_pipeline` is a Galician model originally trained by fpuentes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_galician_pipeline_gl_5.5.1_3.0_1743250362388.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_galician_pipeline_gl_5.5.1_3.0_1743250362388.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_galician_pipeline", lang = "gl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_galician_pipeline", lang = "gl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_galician_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|gl|
|Size:|466.8 MB|

## References

https://huggingface.co/fpuentes/gpt2-galician

## Included Models

- DocumentAssembler
- GPT2Transformer