---
layout: model
title: English gpl_tsdae_e5_base_unsupervised_test_1_pipeline pipeline E5Embeddings from rithwik-db
author: John Snow Labs
name: gpl_tsdae_e5_base_unsupervised_test_1_pipeline
date: 2024-12-17
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

Pretrained E5Embeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpl_tsdae_e5_base_unsupervised_test_1_pipeline` is a English model originally trained by rithwik-db.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpl_tsdae_e5_base_unsupervised_test_1_pipeline_en_5.5.1_3.0_1734398566854.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpl_tsdae_e5_base_unsupervised_test_1_pipeline_en_5.5.1_3.0_1734398566854.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpl_tsdae_e5_base_unsupervised_test_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpl_tsdae_e5_base_unsupervised_test_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpl_tsdae_e5_base_unsupervised_test_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.3 MB|

## References

https://huggingface.co/rithwik-db/gpl_tsdae-e5-base-unsupervised-test-1

## Included Models

- DocumentAssembler
- E5Embeddings