---
layout: model
title: English taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline pipeline BGEEmbeddings from FareedKhan
author: John Snow Labs
name: taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline
date: 2024-12-16
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

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline` is a English model originally trained by FareedKhan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline_en_5.5.1_3.0_1734358236923.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline_en_5.5.1_3.0_1734358236923.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|taylorai_bge_micro_v2_fareedkhan_prime_synthetic_data_2k_10_64_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|55.6 MB|

## References

https://huggingface.co/FareedKhan/TaylorAI_bge-micro-v2_FareedKhan_prime_synthetic_data_2k_10_64

## Included Models

- DocumentAssembler
- BGEEmbeddings