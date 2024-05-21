---
layout: model
title: English mesh_pipeline pipeline BertEmbeddings from Vydiant
author: John Snow Labs
name: mesh_pipeline
date: 2024-05-21
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mesh_pipeline` is a English model originally trained by Vydiant.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mesh_pipeline_en_5.2.4_3.0_1716256848533.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mesh_pipeline_en_5.2.4_3.0_1716256848533.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('mesh_pipeline', lang = 'en')
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline('mesh_pipeline', lang = 'en')
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mesh_pipeline|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|403.1 MB|

## References

https://huggingface.co/Vydiant/mesh-pipeline