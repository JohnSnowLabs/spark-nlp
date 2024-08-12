---
layout: model
title: Galician nos_d2t_galician_pipeline pipeline T5Transformer from proxectonos
author: John Snow Labs
name: nos_d2t_galician_pipeline
date: 2024-08-12
tags: [gl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: gl
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nos_d2t_galician_pipeline` is a Galician model originally trained by proxectonos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nos_d2t_galician_pipeline_gl_5.4.2_3.0_1723438938744.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nos_d2t_galician_pipeline_gl_5.4.2_3.0_1723438938744.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nos_d2t_galician_pipeline", lang = "gl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nos_d2t_galician_pipeline", lang = "gl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nos_d2t_galician_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|gl|
|Size:|2.2 GB|

## References

https://huggingface.co/proxectonos/Nos_D2T-gl

## Included Models

- DocumentAssembler
- T5Transformer