---
layout: model
title: English bge_small_bioasq_1epoch_batch32_pipeline pipeline BGEEmbeddings from juanpablomesa
author: John Snow Labs
name: bge_small_bioasq_1epoch_batch32_pipeline
date: 2024-12-18
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bge_small_bioasq_1epoch_batch32_pipeline` is a English model originally trained by juanpablomesa.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_small_bioasq_1epoch_batch32_pipeline_en_5.5.1_3.0_1734564034634.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_small_bioasq_1epoch_batch32_pipeline_en_5.5.1_3.0_1734564034634.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bge_small_bioasq_1epoch_batch32_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bge_small_bioasq_1epoch_batch32_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_small_bioasq_1epoch_batch32_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|115.7 MB|

## References

https://huggingface.co/juanpablomesa/bge-small-bioasq-1epoch-batch32

## Included Models

- DocumentAssembler
- BGEEmbeddings