---
layout: model
title: English albert_base_v2_2_contract_sections_classification_v4_pipeline pipeline AlbertForSequenceClassification from marcelovidigal
author: John Snow Labs
name: albert_base_v2_2_contract_sections_classification_v4_pipeline
date: 2025-03-27
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_base_v2_2_contract_sections_classification_v4_pipeline` is a English model originally trained by marcelovidigal.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_v2_2_contract_sections_classification_v4_pipeline_en_5.5.1_3.0_1743073666664.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_base_v2_2_contract_sections_classification_v4_pipeline_en_5.5.1_3.0_1743073666664.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_base_v2_2_contract_sections_classification_v4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_base_v2_2_contract_sections_classification_v4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_base_v2_2_contract_sections_classification_v4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|44.3 MB|

## References

https://huggingface.co/marcelovidigal/albert-base-v2-2-contract-sections-classification-v4

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification