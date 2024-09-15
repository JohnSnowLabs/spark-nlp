---
layout: model
title: English all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline pipeline MPNetEmbeddings from luiz-and-robert-thesis
author: John Snow Labs
name: all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline
date: 2024-09-07
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline` is a English model originally trained by luiz-and-robert-thesis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline_en_5.5.0_3.0_1725703181190.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline_en_5.5.0_3.0_1725703181190.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|all_mpnet_base_newtriplets_v2_lr_2e_7_m_5_e_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.9 MB|

## References

https://huggingface.co/luiz-and-robert-thesis/all-mpnet-base-newtriplets-v2-lr-2e-7-m-5-e-3

## Included Models

- DocumentAssembler
- MPNetEmbeddings