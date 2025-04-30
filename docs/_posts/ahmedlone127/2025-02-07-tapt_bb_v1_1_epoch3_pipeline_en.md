---
layout: model
title: English tapt_bb_v1_1_epoch3_pipeline pipeline BertEmbeddings from hyoo14
author: John Snow Labs
name: tapt_bb_v1_1_epoch3_pipeline
date: 2025-02-07
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tapt_bb_v1_1_epoch3_pipeline` is a English model originally trained by hyoo14.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tapt_bb_v1_1_epoch3_pipeline_en_5.5.1_3.0_1738967383503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tapt_bb_v1_1_epoch3_pipeline_en_5.5.1_3.0_1738967383503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tapt_bb_v1_1_epoch3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tapt_bb_v1_1_epoch3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tapt_bb_v1_1_epoch3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.1 MB|

## References

https://huggingface.co/hyoo14/TAPT_BB-v1.1_epoch3

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings