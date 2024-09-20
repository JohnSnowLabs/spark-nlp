---
layout: model
title: Persian albert_persian_poetry_pipeline pipeline AlbertEmbeddings from mitra-mir
author: John Snow Labs
name: albert_persian_poetry_pipeline
date: 2024-09-02
tags: [fa, open_source, pipeline, onnx]
task: Embeddings
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_persian_poetry_pipeline` is a Persian model originally trained by mitra-mir.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_persian_poetry_pipeline_fa_5.5.0_3.0_1725306635075.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_persian_poetry_pipeline_fa_5.5.0_3.0_1725306635075.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_persian_poetry_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_persian_poetry_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_persian_poetry_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|41.9 MB|

## References

https://huggingface.co/mitra-mir/ALBERT-Persian-Poetry

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings