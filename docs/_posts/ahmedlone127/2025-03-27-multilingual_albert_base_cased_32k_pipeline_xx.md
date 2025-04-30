---
layout: model
title: Multilingual multilingual_albert_base_cased_32k_pipeline pipeline AlbertEmbeddings from cservan
author: John Snow Labs
name: multilingual_albert_base_cased_32k_pipeline
date: 2025-03-27
tags: [xx, open_source, pipeline, onnx]
task: Embeddings
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilingual_albert_base_cased_32k_pipeline` is a Multilingual model originally trained by cservan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilingual_albert_base_cased_32k_pipeline_xx_5.5.1_3.0_1743102941409.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilingual_albert_base_cased_32k_pipeline_xx_5.5.1_3.0_1743102941409.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multilingual_albert_base_cased_32k_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multilingual_albert_base_cased_32k_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multilingual_albert_base_cased_32k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|27.5 MB|

## References

https://huggingface.co/cservan/multilingual-albert-base-cased-32k

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings