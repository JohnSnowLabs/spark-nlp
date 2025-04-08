---
layout: model
title: English one_cdk3_m0_5_e5_b128_l6_pipeline pipeline E5Embeddings from swardiantara
author: John Snow Labs
name: one_cdk3_m0_5_e5_b128_l6_pipeline
date: 2025-04-07
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

Pretrained E5Embeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`one_cdk3_m0_5_e5_b128_l6_pipeline` is a English model originally trained by swardiantara.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/one_cdk3_m0_5_e5_b128_l6_pipeline_en_5.5.1_3.0_1744060011945.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/one_cdk3_m0_5_e5_b128_l6_pipeline_en_5.5.1_3.0_1744060011945.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("one_cdk3_m0_5_e5_b128_l6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("one_cdk3_m0_5_e5_b128_l6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|one_cdk3_m0_5_e5_b128_l6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|84.1 MB|

## References

https://huggingface.co/swardiantara/one-cdk3-m0.5-e5-b128-L6

## Included Models

- DocumentAssembler
- E5Embeddings