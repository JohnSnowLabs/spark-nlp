---
layout: model
title: English mpnet_base_nq_cgist_gt_pipeline pipeline MPNetEmbeddings from tomaarsen
author: John Snow Labs
name: mpnet_base_nq_cgist_gt_pipeline
date: 2024-12-14
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mpnet_base_nq_cgist_gt_pipeline` is a English model originally trained by tomaarsen.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mpnet_base_nq_cgist_gt_pipeline_en_5.5.1_3.0_1734210122601.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mpnet_base_nq_cgist_gt_pipeline_en_5.5.1_3.0_1734210122601.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mpnet_base_nq_cgist_gt_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mpnet_base_nq_cgist_gt_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mpnet_base_nq_cgist_gt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|404.9 MB|

## References

https://huggingface.co/tomaarsen/mpnet-base-nq-cgist-gt

## Included Models

- DocumentAssembler
- MPNetEmbeddings