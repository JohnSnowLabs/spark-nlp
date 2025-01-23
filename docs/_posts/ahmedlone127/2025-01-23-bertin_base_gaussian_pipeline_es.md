---
layout: model
title: Castilian, Spanish bertin_base_gaussian_pipeline pipeline RoBertaEmbeddings from bertin-project
author: John Snow Labs
name: bertin_base_gaussian_pipeline
date: 2025-01-23
tags: [es, open_source, pipeline, onnx]
task: Embeddings
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertin_base_gaussian_pipeline` is a Castilian, Spanish model originally trained by bertin-project.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertin_base_gaussian_pipeline_es_5.5.1_3.0_1737643926774.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertin_base_gaussian_pipeline_es_5.5.1_3.0_1737643926774.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertin_base_gaussian_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertin_base_gaussian_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertin_base_gaussian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|231.8 MB|

## References

https://huggingface.co/bertin-project/bertin-base-gaussian

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings