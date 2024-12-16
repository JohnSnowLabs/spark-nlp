---
layout: model
title: English bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline pipeline T5Transformer from MittyN
author: John Snow Labs
name: bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline
date: 2024-12-16
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline` is a English model originally trained by MittyN.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline_en_5.5.1_3.0_1734331838176.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline_en_5.5.1_3.0_1734331838176.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bslm_entity_extraction_mt5_base_include_desc_normalized_tr243k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|2.3 GB|

## References

https://huggingface.co/MittyN/bslm-entity-extraction-mt5-base-include-desc-normalized-tr243k

## Included Models

- DocumentAssembler
- T5Transformer