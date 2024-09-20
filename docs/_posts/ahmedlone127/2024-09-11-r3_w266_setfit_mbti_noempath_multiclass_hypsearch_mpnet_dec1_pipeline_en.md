---
layout: model
title: English r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline pipeline MPNetEmbeddings from shrinivasbjoshi
author: John Snow Labs
name: r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline
date: 2024-09-11
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline` is a English model originally trained by shrinivasbjoshi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline_en_5.5.0_3.0_1726054801614.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline_en_5.5.0_3.0_1726054801614.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|r3_w266_setfit_mbti_noempath_multiclass_hypsearch_mpnet_dec1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.3 MB|

## References

https://huggingface.co/shrinivasbjoshi/r3-w266-setfit-mbti-noempath-multiclass-hypsearch-mpnet-dec1

## Included Models

- DocumentAssembler
- MPNetEmbeddings