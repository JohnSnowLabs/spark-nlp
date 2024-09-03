---
layout: model
title: Arabic albert_base_arabic_pipeline pipeline AlbertEmbeddings from asafaya
author: John Snow Labs
name: albert_base_arabic_pipeline
date: 2024-09-02
tags: [ar, open_source, pipeline, onnx]
task: Embeddings
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_base_arabic_pipeline` is a Arabic model originally trained by asafaya.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_arabic_pipeline_ar_5.5.0_3.0_1725306775483.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_base_arabic_pipeline_ar_5.5.0_3.0_1725306775483.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_base_arabic_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_base_arabic_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_base_arabic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|42.0 MB|

## References

https://huggingface.co/asafaya/albert-base-arabic

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings