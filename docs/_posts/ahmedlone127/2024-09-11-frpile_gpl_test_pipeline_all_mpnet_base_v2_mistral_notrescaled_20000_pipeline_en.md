---
layout: model
title: English frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline pipeline MPNetEmbeddings from DragosGorduza
author: John Snow Labs
name: frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline` is a English model originally trained by DragosGorduza.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline_en_5.5.0_3.0_1726034339824.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline_en_5.5.0_3.0_1726034339824.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|frpile_gpl_test_pipeline_all_mpnet_base_v2_mistral_notrescaled_20000_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/DragosGorduza/FRPile_GPL_test_pipeline_all-mpnet-base-v2-MISTRAL-notrescaled_20000

## Included Models

- DocumentAssembler
- MPNetEmbeddings