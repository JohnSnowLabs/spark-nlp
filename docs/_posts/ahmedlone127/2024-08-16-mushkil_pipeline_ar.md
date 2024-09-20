---
layout: model
title: Arabic mushkil_pipeline pipeline T5Transformer from riotu-lab
author: John Snow Labs
name: mushkil_pipeline
date: 2024-08-16
tags: [ar, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ar
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mushkil_pipeline` is a Arabic model originally trained by riotu-lab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mushkil_pipeline_ar_5.4.2_3.0_1723811125688.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mushkil_pipeline_ar_5.4.2_3.0_1723811125688.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mushkil_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mushkil_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mushkil_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|1.5 GB|

## References

https://huggingface.co/riotu-lab/mushkil

## Included Models

- DocumentAssembler
- T5Transformer