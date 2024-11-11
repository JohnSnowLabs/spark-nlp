---
layout: model
title: English damage_trigger_effect_2024_11_06_13_00_pipeline pipeline BertForTokenClassification from Lolimorimorf
author: John Snow Labs
name: damage_trigger_effect_2024_11_06_13_00_pipeline
date: 2024-11-10
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`damage_trigger_effect_2024_11_06_13_00_pipeline` is a English model originally trained by Lolimorimorf.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/damage_trigger_effect_2024_11_06_13_00_pipeline_en_5.5.1_3.0_1731279697514.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/damage_trigger_effect_2024_11_06_13_00_pipeline_en_5.5.1_3.0_1731279697514.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("damage_trigger_effect_2024_11_06_13_00_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("damage_trigger_effect_2024_11_06_13_00_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|damage_trigger_effect_2024_11_06_13_00_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|665.1 MB|

## References

https://huggingface.co/Lolimorimorf/damage_trigger_effect_2024-11-06_13_00

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification