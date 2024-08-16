---
layout: model
title: Arabic arat5_coda_pipeline pipeline T5Transformer from CAMeL-Lab
author: John Snow Labs
name: arat5_coda_pipeline
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arat5_coda_pipeline` is a Arabic model originally trained by CAMeL-Lab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arat5_coda_pipeline_ar_5.4.2_3.0_1723817228930.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arat5_coda_pipeline_ar_5.4.2_3.0_1723817228930.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arat5_coda_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arat5_coda_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arat5_coda_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|1.4 GB|

## References

https://huggingface.co/CAMeL-Lab/arat5-coda

## Included Models

- DocumentAssembler
- T5Transformer